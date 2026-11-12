import { ModelSelector } from "./ModelSelector";
import type { ModelInfo } from "../api/types";

interface ControlsPanelProps {
  models: ModelInfo[];
  model: string;
  conf: number;
  iou: number;
  imgsz: number;
  onModel: (value: string) => void;
  onConf: (value: number) => void;
  onIou: (value: number) => void;
  onSize: (value: number) => void;
  hideHealthy: boolean;
  onHideHealthy: (value: boolean) => void;
}

export function ControlsPanel(props: ControlsPanelProps) {
  return (
    <section className="controls" aria-label="Điều khiển nhận diện">
      <h2>Điều khiển</h2>
      <ModelSelector models={props.models} value={props.model} onChange={props.onModel} />
      <label>Độ tin cậy <input type="range" min="0.05" max="1" step="0.05" value={props.conf} onChange={(event) => props.onConf(Number(event.target.value))} /><output>{props.conf.toFixed(2)}</output></label>
      <p className="control-hint">Khuyến nghị 0.15–0.30 khi thử ảnh thực tế. Ngưỡng cao chỉ giữ các dự đoán rất chắc chắn.</p>
      <label>IoU (NMS) <input type="range" min="0" max="1" step="0.05" value={props.iou} onChange={(event) => props.onIou(Number(event.target.value))} /><output>{props.iou.toFixed(2)}</output></label>
      <label>Kích thước ảnh <select value={props.imgsz} onChange={(event) => props.onSize(Number(event.target.value))}><option value={320}>320</option><option value={416}>416</option><option value={640}>640</option></select></label>
      <label className="checkbox-control"><input type="checkbox" checked={props.hideHealthy} onChange={(event) => props.onHideHealthy(event.target.checked)} /> Ẩn lá khỏe</label>
    </section>
  );
}
