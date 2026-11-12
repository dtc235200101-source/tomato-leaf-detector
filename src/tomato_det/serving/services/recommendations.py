"""Conservative, class-specific academic follow-up guidance."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

from tomato_det.models.base import Detection


@dataclass(frozen=True, slots=True)
class Recommendation:
    class_id: int
    class_name: str
    title: str
    actions: tuple[str, ...]
    caution: str


_GUIDANCE: dict[str, tuple[str, tuple[str, ...], str]] = {
    "early_blight": (
        "Úa sớm (đốm vòng)",
        (
            "Tỉa bỏ lá có triệu chứng nặng và vệ sinh tàn dư.",
            "Tưới vào gốc, giữ tán thông thoáng và luân canh cây trồng.",
        ),
        "Nếu cần thuốc, hỏi cán bộ khuyến nông về hoạt chất được phép tại địa phương.",
    ),
    "septoria_leaf_spot": (
        "Đốm lá Septoria",
        (
            "Loại bỏ lá nhiễm sớm, khử trùng dụng cụ sau khi tỉa.",
            "Giảm ẩm trên lá; không tưới phun lên tán và duy trì khoảng cách cây.",
        ),
        "Theo dõi lá non định kỳ vì bệnh thường lan từ lá già lên trên.",
    ),
    "bacterial_spot": (
        "Đốm vi khuẩn",
        (
            "Cách ly cây có triệu chứng, hạn chế chạm tay hoặc dụng cụ khi lá ướt.",
            "Dùng hạt/cây giống sạch bệnh, vệ sinh khu vực và tránh tưới phun.",
        ),
        "Không tự pha hoặc tăng liều chế phẩm; tham khảo hướng dẫn chuyên môn địa phương.",
    ),
    "late_blight": (
        "Mốc sương / úa muộn",
        (
            "Kiểm tra toàn ruộng ngay, loại bỏ phần cây nhiễm nặng để giảm nguồn lây.",
            "Giảm thời gian lá ướt, tăng thông thoáng và không để tàn dư bệnh ngoài ruộng.",
        ),
        "Bệnh có thể lan nhanh khi ẩm mát; cần liên hệ cán bộ bảo vệ thực vật sớm.",
    ),
    "mosaic_virus": (
        "Virus khảm",
        (
            "Nhổ và cách ly cây nghi nhiễm để hạn chế lây lan cơ học.",
            "Vệ sinh tay, dụng cụ và kiểm soát côn trùng môi giới theo hướng dẫn địa phương.",
        ),
        "Virus không được xử lý bằng thuốc trừ nấm; nên xác nhận bằng chuyên gia hoặc xét nghiệm khi cần.",
    ),
    "yellow_virus": (
        "Virus vàng lá",
        (
            "Cách ly hoặc loại bỏ cây nghi nhiễm; kiểm tra cây lân cận.",
            "Dùng lưới, giống sạch bệnh và quản lý côn trùng môi giới theo IPM.",
        ),
        "Cần xác nhận thực địa vì vàng lá cũng có thể do dinh dưỡng hoặc stress.",
    ),
    "leaf_mold": (
        "Mốc lá",
        (
            "Tỉa lá bệnh, tăng thông gió và giảm ẩm trong tán/nhà màng.",
            "Tưới vào gốc, tránh để nước đọng trên lá và dọn sạch tàn dư.",
        ),
        "Theo dõi mặt dưới lá vì triệu chứng sớm thường xuất hiện tại đó.",
    ),
    "spider_mites": (
        "Nhện hai chấm",
        (
            "Kiểm tra mặt dưới lá, loại bỏ ổ nhện nặng và giảm bụi khô quanh vườn.",
            "Bảo tồn thiên địch; chỉ dùng biện pháp kiểm soát phù hợp sau khi xác định mật số.",
        ),
        "Tránh dùng thuốc phổ rộng tùy tiện vì có thể làm mất thiên địch và bùng phát nhện.",
    ),
    "healthy": (
        "Lá khỏe",
        (
            "Duy trì tưới, dinh dưỡng và thông thoáng phù hợp.",
            "Tiếp tục theo dõi định kỳ, đặc biệt sau mưa hoặc khi độ ẩm cao.",
        ),
        "Đây là nhận định từ ảnh, không thay thế kiểm tra trực tiếp.",
    ),
}


def recommendations_for(detections: list[Detection]) -> list[Recommendation]:
    """Return one deterministic recommendation per detected class."""
    recommendations: list[Recommendation] = []
    seen: set[str] = set()
    for detection in detections:
        if detection.class_name in seen:
            continue
        seen.add(detection.class_name)
        title, actions, caution = _GUIDANCE[detection.class_name]
        recommendations.append(
            Recommendation(detection.class_id, detection.class_name, title, actions, caution)
        )
    return recommendations
