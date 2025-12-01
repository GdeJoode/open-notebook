import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker deployment
  output: "standalone",

  // Exclude pdfjs-dist from server-side bundling (it's browser-only)
  serverExternalPackages: ["pdfjs-dist", "canvas"],

  // Configure webpack to handle pdfjs-dist properly
  webpack: (config, { isServer }) => {
    // For client-side, provide empty fallbacks for Node.js modules
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        canvas: false,
        fs: false,
        path: false,
      };
    }

    // Ignore canvas module entirely (it's optional for pdfjs-dist)
    config.resolve.alias = {
      ...config.resolve.alias,
      canvas: false,
    };

    // Exclude legacy build which tries to use canvas
    config.externals = config.externals || [];
    if (Array.isArray(config.externals)) {
      config.externals.push({
        canvas: "commonjs canvas",
      });
    }

    return config;
  },
};

export default nextConfig;
