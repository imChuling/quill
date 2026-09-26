import { defineConfig } from "vite"

// 官方要求:OAuth redirect 必须用 127.0.0.1(非 localhost),与注册的 URI 严格一致
export default defineConfig({
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
})
