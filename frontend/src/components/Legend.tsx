import type { ClassInfo } from "../api/types";

export function Legend({ classes }: { classes: ClassInfo[] }) {
  return (
    <section className="legend-block" aria-label="Chú giải lớp">
      <h2>Chú giải lớp</h2>
      <ul className="legend">{classes.map((item) => <li key={item.slug}><i style={{ backgroundColor: item.color }} aria-hidden="true" />{item.name_vi}</li>)}</ul>
    </section>
  );
}
