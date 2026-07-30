// eslint-config-next 16 ships a native flat config array, so it is spread
// directly. The FlatCompat bridge that older setups used now throws on it.
import next from "eslint-config-next";

const config = [
  ...next,
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts"] },
];

export default config;
