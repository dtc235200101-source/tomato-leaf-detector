import { useEffect, useState } from "react";

import { getClasses, getModels } from "./api/client";
import type { ClassInfo, ModelInfo } from "./api/types";
import { AboutPage } from "./pages/AboutPage";
import { DatasetPage } from "./pages/DatasetPage";
import { LibraryPage } from "./pages/LibraryPage";
import { LivePage } from "./pages/LivePage";
import { UploadPage } from "./pages/UploadPage";

type Page = "upload" | "live" | "dataset" | "library" | "about";

const NAV_ITEMS: { id: Page; label: string; icon: string }[] = [
  { id: "upload", label: "Nhận dạng qua ảnh", icon: "🖼️" },
  { id: "live", label: "Camera trực tiếp", icon: "📷" },
  { id: "dataset", label: "Dữ liệu dataset", icon: "🗂️" },
  { id: "library", label: "Thư viện bệnh", icon: "📖" },
  { id: "about", label: "Giới thiệu", icon: "ℹ️" },
];

export default function App() {
  const [page, setPage] = useState<Page>("upload");
  const [classes, setClasses] = useState<ClassInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([getClasses(), getModels()])
      .then(([nextClasses, nextModels]) => {
        setClasses(nextClasses);
        setModels(nextModels);
      })
      .catch(() => setError("Không kết nối được backend local. Kiểm tra server đã chạy chưa."));
  }, []);

  return (
    <div className="app-shell">
      <header>
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 64 64" width="38" height="38" aria-hidden="true">
              <defs>
                <linearGradient id="lgTomato" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="#ff6b6b" /><stop offset="1" stopColor="#e73838" />
                </linearGradient>
                <linearGradient id="lgLeaf" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="#7bd88f" /><stop offset="1" stopColor="#2a9d8f" />
                </linearGradient>
              </defs>
              <circle cx="32" cy="38" r="20" fill="url(#lgTomato)" />
              <path d="M32 18c-2-6 2-10 6-12-1 6-2 8-6 12z" fill="url(#lgLeaf)" />
              <path d="M32 20c-4-3-9-3-12-1 4 3 8 3 12 1z" fill="url(#lgLeaf)" />
              <path d="M32 20c4-3 9-3 12-1-4 3-8 3-12 1z" fill="url(#lgLeaf)" />
              <circle cx="24" cy="32" r="5" fill="#ffffff" opacity=".35" />
              <path d="M22 47c2 3 6 4 9 2" stroke="#fff" strokeWidth="2.5" fill="none" strokeLinecap="round" opacity=".6" />
            </svg>
          </div>
          <div>
            <strong><span className="grad">Phát hiện bệnh lá cà chua</span></strong>
            <small>trong ảnh thực địa · YOLOv11 — nhận dạng, khoanh vùng &amp; hướng dẫn xử lý</small>
          </div>
        </div>
        <nav aria-label="Điều hướng">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={page === item.id ? "active" : ""}
              aria-current={page === item.id ? "page" : undefined}
              onClick={() => setPage(item.id)}
            >
              <span aria-hidden="true">{item.icon}</span> {item.label}
            </button>
          ))}
        </nav>
      </header>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {page === "upload" ? (
        <UploadPage classes={classes} models={models} />
      ) : page === "live" ? (
        <LivePage classes={classes} models={models} />
      ) : page === "dataset" ? (
        <DatasetPage classes={classes} />
      ) : page === "library" ? (
        <LibraryPage classes={classes} />
      ) : (
        <AboutPage />
      )}
    </div>
  );
}
