import { useEffect, useMemo, useState } from "react";

import { datasetImage, getDatasetSplits, type SplitSummary } from "../api/client";
import type { ClassInfo } from "../api/types";

const GROUP_VI: Record<string, string> = {
  fungal_oomycete: "nấm",
  bacterial: "vi khuẩn",
  viral: "virus",
  pest: "côn trùng",
  healthy: "khỏe",
};

export function DatasetPage({ classes }: { classes: ClassInfo[] }) {
  const [splits, setSplits] = useState<SplitSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<string>("val");
  const [zoom, setZoom] = useState<string | null>(null);

  useEffect(() => {
    getDatasetSplits()
      .then(setSplits)
      .catch(() => setError("Không đọc được dữ liệu dataset — backend chưa chạy."));
  }, []);

  const current = useMemo(() => splits.find((item) => item.name === active) ?? null, [splits, active]);

  const closeZoom = () => setZoom(null);

  return (
    <main className="page-grid dataset-page">
      <section className="panel">
        <h1>Dữ liệu dataset — PlantDoc (tomato)</h1>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="stats-grid">
          {splits.length === 0 && !error && <p>Đang tải thống kê…</p>}
          {splits.map((item) => (
            <div key={item.name} className="stat-card">
              <b>{item.count}</b>
              <span>ảnh {item.name}</span>
            </div>
          ))}
          <div className="stat-card">
            <b>{classes.length}</b>
            <span>lớp bệnh</span>
          </div>
        </div>

        <div className="split-tabs" role="tablist">
          {splits.map((item) => (
            <button
              key={item.name}
              type="button"
              className={active === item.name ? "active" : ""}
              onClick={() => setActive(item.name)}
            >
              {item.name} ({item.count})
            </button>
          ))}
        </div>
        <p className="hint">Nhấp vào ảnh để xem lớn — ảnh đọc trực tiếp qua API backend.</p>

        <div className="gallery">
          {current?.samples.map((imageId) => {
            const src = datasetImage(current.name, imageId);
            return (
              <figure key={imageId} title={imageId}>
                <img
                  src={src}
                  alt={imageId}
                  loading="lazy"
                  onClick={() => setZoom(src)}
                  onError={(event) => {
                    const target = event.currentTarget;
                    target.parentElement!.classList.add("broken");
                    target.remove();
                  }}
                />
                <figcaption>{(imageId.split("/").pop() ?? "").slice(0, 26)}</figcaption>
              </figure>
            );
          })}
        </div>

        <h2>🏷️ Các lớp bệnh (configs/classes.yaml)</h2>
        <table className="class-table">
          <thead>
            <tr><th>ID</th><th>Lớp</th><th>Tên tiếng Việt</th><th>Nhóm</th></tr>
          </thead>
          <tbody>
            {classes.map((item) => (
              <tr key={item.slug}>
                <td>{item.id}</td>
                <td>{item.slug}</td>
                <td>
                  <i className="dot" style={{ backgroundColor: item.color }} />
                  {item.name_vi}
                </td>
                <td><span className="tag">{GROUP_VI[item.group] ?? item.group}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {zoom && (
        <div className="lightbox" role="dialog" onClick={closeZoom}>
          <button type="button" aria-label="Đóng">✕</button>
          <img src={zoom} alt="phóng to" onClick={(event) => event.stopPropagation()} />
        </div>
      )}
    </main>
  );
}
