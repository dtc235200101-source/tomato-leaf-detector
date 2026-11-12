import type { Recommendation } from "../api/types";

export function RecommendationList({ recommendations }: { recommendations: Recommendation[] }) {
  if (recommendations.length === 0) return null;
  return (
    <section className="advice-box" aria-label="Hướng xử lý đề xuất">
      <b>Hướng xử lý đề xuất theo nhãn phát hiện:</b>
      {recommendations.map((item) => (
        <article key={item.class_name} className="advice-item">
          <strong>{item.title}</strong>
          <ul>{item.actions.map((action) => <li key={action}>{action}</li>)}</ul>
          <small>{item.caution}</small>
        </article>
      ))}
      <em>⚠️ Kết quả AI chỉ để tham khảo học thuật; xác nhận thực địa trước khi can thiệp.</em>
    </section>
  );
}
