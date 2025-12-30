import { z } from "zod";
import { publicProcedure } from "../../../create-context";

export default publicProcedure
  .input(z.object({
    width: z.number(),
    height: z.number()
  }))
  .query(({ input }) => {
    const totalPixels = input.width * input.height;
    const bitsPerPixel = 3;
    const totalBits = totalPixels * bitsPerPixel;
    
    const headerBits = 128;
    const crcBits = 32;
    
    const maxPayloadBits = totalBits - headerBits - crcBits;
    const maxPayloadBytes = Math.floor(maxPayloadBits / 8);

    return {
      width: input.width,
      height: input.height,
      totalPixels,
      totalBits,
      headerBits,
      crcBits,
      maxPayloadBytes,
      maxPayloadBits
    };
  });
