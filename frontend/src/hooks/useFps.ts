import { useCallback, useRef, useState } from "react";

export function useFps() {
  const previous = useRef<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);

  const markFrame = useCallback(() => {
    const now = performance.now();
    if (previous.current !== null && now > previous.current) setFps(1000 / (now - previous.current));
    previous.current = now;
  }, []);

  return { fps, markFrame };
}
