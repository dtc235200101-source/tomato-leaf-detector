const STATUS_LABELS: Record<string, string> = {
  idle: "chưa kết nối",
  connecting: "đang kết nối",
  connected: "đã kết nối",
  closed: "đã đóng",
  error: "lỗi",
};

export function StatsBar({ fps, latency, status }: { fps: number | null; latency: number | null; status: string }) {
  return (
    <div className="stats" role="status">
      <span><span className="stat-label">Kết nối:</span> <strong>{STATUS_LABELS[status] ?? status}</strong></span>
      <span><span className="stat-label">FPS:</span> <strong>{fps === null ? "—" : fps.toFixed(1)}</strong></span>
      <span><span className="stat-label">Suy luận:</span> <strong>{latency === null ? "—" : `${latency.toFixed(1)} ms`}</strong></span>
    </div>
  );
}
