const fallbackColors = ["#ff8c42", "#62c370", "#5dade2", "#f4d35e", "#c084fc"];

export function colorForClass(className: string, colors: Record<string, string>): string {
  return colors[className] ?? fallbackColors[Array.from(className).reduce((sum, char) => sum + char.charCodeAt(0), 0) % fallbackColors.length];
}
