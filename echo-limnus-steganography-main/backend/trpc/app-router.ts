import { createTRPCRouter } from "./create-context";
import hiRoute from "./routes/example/hi/route";
import lsbDecodeRoute from "./routes/lsb/decode/route";
import lsbEncodeRoute from "./routes/lsb/encode/route";
import lsbCapacityRoute from "./routes/lsb/capacity/route";

export const appRouter = createTRPCRouter({
  example: createTRPCRouter({
    hi: hiRoute,
  }),
  lsb: createTRPCRouter({
    decode: lsbDecodeRoute,
    encode: lsbEncodeRoute,
    capacity: lsbCapacityRoute,
  }),
});

export type AppRouter = typeof appRouter;
