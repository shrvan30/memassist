/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone keeps the docker image to the server plus the files it actually
  // needs, instead of the whole node_modules tree.
  output: "standalone",
  reactStrictMode: true,
};
export default nextConfig;
