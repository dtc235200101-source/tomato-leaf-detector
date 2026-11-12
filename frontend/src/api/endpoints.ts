/**
 * Resolves API endpoints for both the local all-in-one server and a separately
 * deployed frontend. Leave VITE_API_BASE_URL empty for the local default.
 */
const configuredBase = (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");

export function apiUrl(path: string): string {
  if (!path.startsWith("/")) throw new Error("API paths must start with '/'");
  return configuredBase ? `${configuredBase}${path}` : path;
}

export function websocketApiUrl(path: string): string {
  if (!path.startsWith("/")) throw new Error("WebSocket paths must start with '/'");
  if (!configuredBase) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}${path}`;
  }

  const endpoint = new URL(configuredBase);
  endpoint.protocol = endpoint.protocol === "https:" ? "wss:" : "ws:";
  endpoint.pathname = `${endpoint.pathname.replace(/\/$/, "")}${path}`;
  endpoint.search = "";
  endpoint.hash = "";
  return endpoint.toString();
}
