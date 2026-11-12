import type { ModelInfo } from "../api/types";

export function ModelSelector({ models, value, onChange }: { models: ModelInfo[]; value: string; onChange: (value: string) => void }) {
  return (
    <label>
      Mô hình
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {models.map((model) => (
          <option key={model.model_id} value={model.model_id} disabled={!model.available}>
            {model.display_name}{model.available ? "" : " — chưa sẵn sàng"}
          </option>
        ))}
      </select>
    </label>
  );
}
