import { useRef } from "react";

export function UploadDropzone({ onFile }: { onFile: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  return <section className="upload-dropzone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) onFile(file); }}><input ref={inputRef} type="file" accept="image/*" hidden onChange={(event) => { const file = event.target.files?.[0]; if (file) onFile(file); }} /><p>Kéo ảnh lá vào đây hoặc chọn ảnh từ máy.</p><button type="button" className="secondary" onClick={() => inputRef.current?.click()}>Chọn ảnh</button></section>;
}
