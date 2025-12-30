export interface LSBHeader {
  magic: string;
  version: number;
  payloadLength: number;
  flags: number;
}

export interface LSBDecodeResult {
  header: LSBHeader;
  payloadBase64: string;
  payloadText: string;
  crc32?: string;
  isValid: boolean;
  error?: string;
}

function readBitsFromPixel(r: number, g: number, b: number): [number, number, number] {
  return [(r >> 7) & 1, (g >> 7) & 1, (b >> 7) & 1];
}

function bitsToBytes(bits: number[]): Uint8Array {
  const bytes = new Uint8Array(Math.floor(bits.length / 8));
  for (let i = 0; i < bytes.length; i++) {
    let byte = 0;
    for (let j = 0; j < 8; j++) {
      byte = (byte << 1) | (bits[i * 8 + j] || 0);
    }
    bytes[i] = byte;
  }
  return bytes;
}

function calculateCRC32(data: Uint8Array): string {
  const polynomial = 0xEDB88320;
  let crc = 0xFFFFFFFF;

  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (crc & 1 ? polynomial : 0);
    }
  }

  return ((crc ^ 0xFFFFFFFF) >>> 0).toString(16).toUpperCase().padStart(8, '0');
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export async function decodeLSBImage(imageUri: string): Promise<LSBDecodeResult> {
  try {
    const response = await fetch(imageUri);
    const blob = await response.blob();
    
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            reject(new Error('Failed to get canvas context'));
            return;
          }

          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const pixels = imageData.data;

          const bits: number[] = [];
          for (let i = 0; i < pixels.length; i += 4) {
            const [rBit, gBit, bBit] = readBitsFromPixel(pixels[i], pixels[i + 1], pixels[i + 2]);
            bits.push(rBit, gBit, bBit);
          }

          const headerBits = bits.slice(0, 128);
          const headerBytes = bitsToBytes(headerBits);

          const magic = String.fromCharCode(...headerBytes.slice(0, 4));
          const version = headerBytes[4];
          const payloadLength = (headerBytes[5] << 24) | (headerBytes[6] << 16) | (headerBytes[7] << 8) | headerBytes[8];
          const flags = headerBytes[9];

          const header: LSBHeader = { magic, version, payloadLength, flags };

          if (magic !== 'LSB1') {
            const legacyResult = decodeLegacyNullTerminated(bits);
            if (legacyResult) {
              resolve(legacyResult);
              return;
            }
            resolve({
              header,
              payloadBase64: '',
              payloadText: '',
              isValid: false,
              error: 'Invalid magic number and no legacy payload found'
            });
            return;
          }

          const payloadStartBit = 128;
          const payloadBitCount = payloadLength * 8;
          const payloadBits = bits.slice(payloadStartBit, payloadStartBit + payloadBitCount);
          const payloadBytes = bitsToBytes(payloadBits);

          const hasCRC = (flags & 0x01) !== 0;
          let crc32: string | undefined;
          let isValid = true;

          if (hasCRC) {
            const crcStartBit = payloadStartBit + payloadBitCount;
            const crcBits = bits.slice(crcStartBit, crcStartBit + 32);
            const crcBytes = bitsToBytes(crcBits);
            const storedCRC = Array.from(crcBytes).map(b => b.toString(16).toUpperCase().padStart(2, '0')).join('');
            const calculatedCRC = calculateCRC32(payloadBytes);
            crc32 = storedCRC;
            isValid = storedCRC === calculatedCRC;
          }

          const payloadBase64 = uint8ArrayToBase64(payloadBytes);
          const payloadText = new TextDecoder().decode(base64ToUint8Array(payloadBase64));

          resolve({
            header,
            payloadBase64,
            payloadText,
            crc32,
            isValid
          });
        } catch (error) {
          reject(error);
        }
      };

      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(blob);
    });
  } catch (error) {
    throw new Error(`Decode failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

function decodeLegacyNullTerminated(bits: number[]): LSBDecodeResult | null {
  const bytes: number[] = [];
  for (let i = 0; i < bits.length; i += 8) {
    if (i + 8 > bits.length) break;
    let byte = 0;
    for (let j = 0; j < 8; j++) {
      byte = (byte << 1) | (bits[i + j] || 0);
    }
    if (byte === 0) break;
    bytes.push(byte);
  }

  if (bytes.length === 0) return null;

  const payloadBytes = new Uint8Array(bytes);
  const payloadText = new TextDecoder().decode(payloadBytes);
  const payloadBase64 = uint8ArrayToBase64(payloadBytes);

  return {
    header: {
      magic: 'LEGACY',
      version: 0,
      payloadLength: bytes.length,
      flags: 0
    },
    payloadBase64,
    payloadText,
    isValid: true
  };
}
