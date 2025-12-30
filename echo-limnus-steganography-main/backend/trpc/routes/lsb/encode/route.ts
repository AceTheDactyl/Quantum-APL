import { z } from "zod";
import { publicProcedure } from "../../../create-context";

function calculateCRC32(data: Uint8Array): Uint8Array {
  const polynomial = 0xEDB88320;
  let crc = 0xFFFFFFFF;

  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? polynomial : 0);
    }
  }

  const finalCRC = (crc ^ 0xFFFFFFFF) >>> 0;
  return new Uint8Array([
    (finalCRC >> 24) & 0xFF,
    (finalCRC >> 16) & 0xFF,
    (finalCRC >> 8) & 0xFF,
    finalCRC & 0xFF
  ]);
}

function bytesToBits(bytes: Uint8Array): number[] {
  const bits: number[] = [];
  for (let i = 0; i < bytes.length; i++) {
    for (let j = 7; j >= 0; j--) {
      bits.push((bytes[i] >> j) & 1);
    }
  }
  return bits;
}

function embedBitsInPixels(pixels: Uint8Array, bits: number[]): Uint8Array {
  const result = new Uint8Array(pixels);
  let bitIndex = 0;

  for (let i = 0; i < result.length && bitIndex < bits.length; i += 4) {
    if (bitIndex < bits.length) {
      result[i] = (result[i] & 0xFE) | bits[bitIndex++];
    }
    if (bitIndex < bits.length) {
      result[i + 1] = (result[i + 1] & 0xFE) | bits[bitIndex++];
    }
    if (bitIndex < bits.length) {
      result[i + 2] = (result[i + 2] & 0xFE) | bits[bitIndex++];
    }
  }

  return result;
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = Buffer.from(base64, 'base64').toString('binary');
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

export default publicProcedure
  .input(z.object({
    coverImageBase64: z.string(),
    width: z.number(),
    height: z.number(),
    payloadBase64: z.string(),
    includeCRC: z.boolean().default(true)
  }))
  .mutation(({ input }) => {
    try {
      const coverBuffer = Buffer.from(input.coverImageBase64, 'base64');
      const pixels = new Uint8Array(coverBuffer);

      const payloadBytes = base64ToUint8Array(input.payloadBase64);
      
      const header = new Uint8Array(16);
      header[0] = 'L'.charCodeAt(0);
      header[1] = 'S'.charCodeAt(0);
      header[2] = 'B'.charCodeAt(0);
      header[3] = '1'.charCodeAt(0);
      header[4] = 1;
      header[5] = (payloadBytes.length >> 24) & 0xFF;
      header[6] = (payloadBytes.length >> 16) & 0xFF;
      header[7] = (payloadBytes.length >> 8) & 0xFF;
      header[8] = payloadBytes.length & 0xFF;
      header[9] = input.includeCRC ? 0x01 : 0x00;

      const headerBits = bytesToBits(header);
      const payloadBits = bytesToBits(payloadBytes);
      
      let allBits = [...headerBits, ...payloadBits];

      if (input.includeCRC) {
        const crcBytes = calculateCRC32(payloadBytes);
        const crcBits = bytesToBits(crcBytes);
        allBits = [...allBits, ...crcBits];
      }

      const capacity = Math.floor((pixels.length / 4) * 3);
      if (allBits.length > capacity) {
        throw new Error(`Payload too large. Need ${allBits.length} bits, capacity is ${capacity} bits`);
      }

      const stegoPixels = embedBitsInPixels(pixels, allBits);
      const stegoBase64 = Buffer.from(stegoPixels).toString('base64');

      return {
        stegoImageBase64: stegoBase64,
        payloadLength: payloadBytes.length,
        bitsUsed: allBits.length,
        capacity
      };
    } catch (error) {
      throw new Error(`Encode failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  });
