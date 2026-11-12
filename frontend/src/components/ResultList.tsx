import type { ClassInfo, Detection } from "../api/types";

export function ResultList({ detections, classes }: { detections: Detection[]; classes: ClassInfo[] }) {
  const meta = new Map(classes.map((item) => [item.slug, item]));
  return (
    <section className="result-list" aria-live="polite">
      <h2>Kết quả ({detections.length})</h2>
      {detections.length === 0 ? <p>Chưa có vùng lá nào được phát hiện.</p> : <ul>{detections.map((item, index) => <li key={`${item.class_name}-${index}`}><i className="result-dot" style={{ backgroundColor: meta.get(item.class_name)?.color ?? "#8a958c" }} aria-hidden="true" /><strong className="result-name">{meta.get(item.class_name)?.name_vi ?? item.class_name}</strong><span>{(item.score * 100).toFixed(1)}%</span></li>)}</ul>}
    </section>
  );
}
