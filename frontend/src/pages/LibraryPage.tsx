import { datasetImage } from "../api/client";
import type { ClassInfo } from "../api/types";
import { useState } from "react";

const DESCRIPTIONS: Record<string, string> = {
  early_blight: "Nấm Alternaria solani. Đốm tròn đồng tâm màu nâu như vầng đá, lan từ lá dưới lên trên.",
  septoria_leaf_spot: "Nấm Septoria lycopersici. Chấm nhỏ tròn 2–3 mm, trung tâm xám viền tím, lá khô rồi rụng.",
  bacterial_spot: "Xanthomonas spp. Đốm tối bóng nước, mép vàng, lá rách như bị đạn bắn.",
  late_blight: "Phytophthora infestans. Vết loang mờ sũng nước, mặt dưới có lớp lông mốc trắng.",
  mosaic_virus: "Tomato mosaic virus. Lá nhổn, xanh vàng lốm đốm khảm, cây lùn.",
  yellow_virus: "TYLCV. Lá vàng xoăn lên, nhỏ lại, hoa rụng, khó ra quả.",
  leaf_mold: "Fulvia fulva. Mặt trên có vệt vàng nhợt, mặt dưới phủ lớp mốc olive xám.",
  spider_mites: "Tetranychus urticae. Lá đầy chấm li ti vàng trắng, mặt dưới có mạng nhện mỏng.",
  healthy: "Không phát hiện triệu chứng bệnh — nhận diện để giảm báo động nhầm.",
};

const EXAMPLES: Record<string, string> = {
  early_blight: "train/1576389-early%20blight%20symptoms.jpg",
  septoria_leaf_spot: "train/3-septoria-leaf-blight-n.jpg",
  bacterial_spot: "train/Bacterial+Spot+-+Tomatoes.jpg",
  late_blight: "train/2017-08-27%2B-%2BLate%2BBlight%2Bon%2BGypsy%2B1.jpg",
  mosaic_virus: "train/3-tomato-mosaic-virus_web.jpg",
  yellow_virus: "train/tomato-plant-virus-tomato-yellow-leaf-curl-virus-tomato-plant-disease-mosaic-virus.jpg",
  leaf_mold: "train/Leaf+mold+tomato+%25282%2529.JPG.jpg",
  spider_mites: "train/SpotSpeckBlightMite-1l4v879.jpg",
  healthy: "train/8.--Virus-Damaged-foliage-at-top-versus-healthy,-non-infected-bottom-foliage.jpg",
};

export function LibraryPage({ classes }: { classes: ClassInfo[] }) {
  const [zoom, setZoom] = useState<{ src: string; title: string } | null>(null);
  const closeZoom = () => setZoom(null);

  return (
    <main className="page-grid library-page">
      <section className="panel">
        <h1>Thư viện bệnh lá cà chua</h1>
        <div className="library-grid">
          {classes.map((item) => {
            const example = EXAMPLES[item.slug];
            return (
              <article key={item.slug} className="library-card">
                {example && (
                  <img
                    className="library-img"
                    src={datasetImage("train", example)}
                    alt={item.name_vi}
                    loading="lazy"
                    onClick={() => setZoom({ src: datasetImage("train", example), title: item.name_vi })}
                    onError={(event) => event.currentTarget.remove()}
                  />
                )}
                <header>
                  <i className="dot" style={{ backgroundColor: item.color }} />
                  <h3>{item.name_vi}</h3>
                  <small>{item.slug}</small>
                </header>
                <p>{DESCRIPTIONS[item.slug] ?? ""}</p>
                {example && <small className="library-note">📸 Bấm ảnh để xem lớn — lấy từ dataset thật</small>}
              </article>
            );
          })}
        </div>
      </section>

      {zoom && (
        <div className="lightbox" role="dialog" onClick={closeZoom}>
          <button type="button" aria-label="Đóng">✕</button>
          <img src={zoom.src} alt={zoom.title} onClick={(event) => event.stopPropagation()} />
          <p>{zoom.title}</p>
        </div>
      )}
    </main>
  );
}
