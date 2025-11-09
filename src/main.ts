import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const shapes: DetectedShape[] = [];
    const width = imageData.width;
    const height = imageData.height;

    // Step 1: Convert to grayscale and threshold
    const binary = this.preprocessImage(imageData);

    // Debug: Log binary image stats
    let blackPixels = 0;
    for (let i = 0; i < binary.length; i++) {
      if (binary[i] === 1) blackPixels++;
    }
    console.log(
      `Binary image stats: ${blackPixels}/${binary.length} black pixels (${(
        (blackPixels / binary.length) *
        100
      ).toFixed(1)}%)`
    );

    // Step 2: Find contours
    const contours = this.findContours(binary, width, height);
    console.log(`Found ${contours.length} contours`);

    // Step 3: Analyze each contour and classify shapes
    for (const contourData of contours) {
      console.log(
        `\nContour ${shapes.length + 1}: ${
          contourData.contour.length
        } points, area: ${contourData.area}`
      );

      // Filter out very small shapes - likely text fragments or noise
      // Real geometric shapes should be at least 100 pixels
      if (contourData.area < 100) {
        console.log(
          "  -> Skipping: area too small (< 100 pixels, likely text/noise)"
        );
        continue;
      }

      if (contourData.contour.length < 4) {
        console.log("  -> Skipping: too few boundary points (< 4)");
        continue;
      }

      const shape = this.analyzeContour(contourData.contour, contourData.area);
      if (shape) {
        console.log(
          `  ✓ Detected ${shape.type.toUpperCase()} with confidence ${(
            shape.confidence * 100
          ).toFixed(1)}%`
        );
        shapes.push(shape);
      } else {
        console.log("  ✗ Could not classify shape");
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  /**
   * Preprocess image: convert to grayscale and apply thresholding
   */
  private preprocessImage(imageData: ImageData): Uint8Array {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const binary = new Uint8Array(width * height);

    // First pass: convert to grayscale
    const grayscale = new Uint8Array(width * height);
    let minGray = 255;
    let maxGray = 0;

    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      grayscale[i] = gray;
      minGray = Math.min(minGray, gray);
      maxGray = Math.max(maxGray, gray);
    }

    // Use Otsu's method for adaptive thresholding
    let threshold = this.calculateOtsuThreshold(grayscale);

    // Fallback: if threshold is at extremes, use middle value
    if (threshold < 10 || threshold > 245) {
      threshold = (minGray + maxGray) / 2;
      console.log(`Using fallback threshold: ${threshold.toFixed(1)}`);
    }

    console.log(
      `Threshold: ${threshold.toFixed(1)}, range: ${minGray}-${maxGray}`
    );

    // Determine if we have dark shapes on light background or vice versa
    // Check edges to determine background color
    let edgeSum = 0;
    let edgeCount = 0;

    // Sample pixels from edges
    for (let x = 0; x < width; x++) {
      edgeSum += grayscale[x]; // Top edge
      edgeSum += grayscale[(height - 1) * width + x]; // Bottom edge
      edgeCount += 2;
    }
    for (let y = 1; y < height - 1; y++) {
      edgeSum += grayscale[y * width]; // Left edge
      edgeSum += grayscale[y * width + width - 1]; // Right edge
      edgeCount += 2;
    }

    const edgeAvg = edgeSum / edgeCount;
    const darkOnLight = edgeAvg > threshold;

    console.log(
      `Edge avg: ${edgeAvg.toFixed(1)}, threshold: ${threshold.toFixed(1)}`
    );
    console.log(
      `Detected: ${
        darkOnLight
          ? "dark shapes on light background"
          : "light shapes on dark background"
      }`
    );

    for (let i = 0; i < width * height; i++) {
      if (darkOnLight) {
        // Dark shapes on light background
        binary[i] = grayscale[i] < threshold ? 1 : 0;
      } else {
        // Light shapes on dark background
        binary[i] = grayscale[i] > threshold ? 1 : 0;
      }
    }

    return binary;
  }

  /**
   * Calculate Otsu's threshold for optimal binarization
   */
  private calculateOtsuThreshold(grayscale: Uint8Array): number {
    const histogram = new Array(256).fill(0);

    // Build histogram
    for (let i = 0; i < grayscale.length; i++) {
      histogram[Math.floor(grayscale[i])]++;
    }

    const total = grayscale.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      sum += i * histogram[i];
    }

    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let maxVariance = 0;
    let threshold = 0;

    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;

      wF = total - wB;
      if (wF === 0) break;

      sumB += t * histogram[t];

      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;

      const variance = wB * wF * Math.pow(mB - mF, 2);

      if (variance > maxVariance) {
        maxVariance = variance;
        threshold = t;
      }
    }

    return threshold;
  }

  /**
   * Find contours in binary image using border following algorithm
   */
  private findContours(
    binary: Uint8Array,
    width: number,
    height: number
  ): Array<{ contour: Point[]; area: number }> {
    const visited = new Uint8Array(width * height);
    const contours: Array<{ contour: Point[]; area: number }> = [];

    // Apply morphological operations to clean up the binary image
    const cleaned = this.morphologicalClean(binary, width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (cleaned[idx] === 1 && visited[idx] === 0) {
          const result = this.traceContour(
            cleaned,
            visited,
            x,
            y,
            width,
            height
          );
          // Very lenient requirements - let shape analysis do the filtering
          if (result.boundary.length > 4 && result.area > 15) {
            contours.push({ contour: result.boundary, area: result.area });
          }
        }
      }
    }

    return contours;
  }

  /**
   * Apply morphological operations to clean up binary image
   */
  private morphologicalClean(
    binary: Uint8Array,
    width: number,
    height: number
  ): Uint8Array {
    // Count black pixels to determine if we should apply morphological operations
    let blackPixelCount = 0;
    for (let i = 0; i < binary.length; i++) {
      if (binary[i] === 1) blackPixelCount++;
    }

    const blackPixelRatio = blackPixelCount / binary.length;
    console.log(`  Black pixel ratio: ${(blackPixelRatio * 100).toFixed(2)}%`);

    // If very few black pixels (< 5%), likely no real shapes - return empty
    // This prevents false positives from noise/text/lines
    if (blackPixelRatio < 0.05) {
      console.log("  Too few foreground pixels - likely no shapes");
      return new Uint8Array(width * height); // Return empty array
    }

    // If we have a reasonable amount of foreground pixels, apply mild morphological operations
    // But only if there's not too much foreground (which would indicate a bad threshold)
    if (blackPixelRatio > 0.7) {
      console.log("  Too many foreground pixels - threshold might be inverted");
      // Don't apply morphological operations, just return original
      return new Uint8Array(binary);
    }

    // Simple morphological opening (erosion followed by dilation)
    // This helps remove small noise while preserving shape boundaries
    const result = new Uint8Array(binary);

    // 3x3 cross kernel for conservative processing
    const kernel = [
      [0, 1, 0],
      [1, 1, 1],
      [0, 1, 0],
    ];

    // First pass: mild erosion to remove noise
    const eroded = new Uint8Array(width * height);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let allNeighbors = true;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            if (kernel[ky + 1][kx + 1] === 1) {
              const idx = (y + ky) * width + (x + kx);
              if (result[idx] === 0) {
                allNeighbors = false;
                break;
              }
            }
          }
          if (!allNeighbors) break;
        }
        eroded[y * width + x] = allNeighbors ? 1 : 0;
      }
    }

    // Second pass: mild dilation to restore shape size
    const dilated = new Uint8Array(width * height);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let hasNeighbor = false;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            if (kernel[ky + 1][kx + 1] === 1) {
              const idx = (y + ky) * width + (x + kx);
              if (eroded[idx] === 1) {
                hasNeighbor = true;
                break;
              }
            }
          }
          if (hasNeighbor) break;
        }
        dilated[y * width + x] = hasNeighbor ? 1 : 0;
      }
    }

    return dilated;
  }

  /**
   * Trace a single contour using flood fill approach to get all pixels in the component
   * Returns an object with both boundary points and total area
   */
  private traceContour(
    binary: Uint8Array,
    visited: Uint8Array,
    startX: number,
    startY: number,
    width: number,
    height: number
  ): { boundary: Point[]; area: number } {
    const allPoints: Point[] = [];
    const stack: Point[] = [{ x: startX, y: startY }];
    const directions = [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];

    while (stack.length > 0) {
      const point = stack.pop()!;
      const idx = point.y * width + point.x;

      if (visited[idx] === 1) continue;
      if (binary[idx] === 0) continue;

      visited[idx] = 1;
      allPoints.push(point);

      // Add neighbors to stack
      for (const [dx, dy] of directions) {
        const nx = point.x + dx;
        const ny = point.y + dy;
        if (
          nx >= 0 &&
          nx < width &&
          ny >= 0 &&
          ny < height &&
          binary[ny * width + nx] === 1 &&
          visited[ny * width + nx] === 0
        ) {
          stack.push({ x: nx, y: ny });
        }
      }
    }

    // Extract boundary points using Moore-Neighbor tracing
    const boundary = this.extractBoundary(allPoints);
    // Return both boundary and the actual filled area (count of all pixels)
    return { boundary, area: allPoints.length };
  }

  /**
   * Extract boundary points from a set of points
   */
  private extractBoundary(points: Point[]): Point[] {
    if (points.length === 0) return [];

    // Create a set for quick lookup
    const pointSet = new Set<string>();
    for (const p of points) {
      pointSet.add(`${p.x},${p.y}`);
    }

    const boundary: Point[] = [];
    const directions = [
      [-1, -1],
      [0, -1],
      [1, -1],
      [-1, 0],
      [1, 0],
      [-1, 1],
      [0, 1],
      [1, 1],
    ];

    for (const point of points) {
      // A point is on the boundary if any neighbor is not in the set
      let isBoundary = false;
      for (const [dx, dy] of directions) {
        const nx = point.x + dx;
        const ny = point.y + dy;
        if (!pointSet.has(`${nx},${ny}`)) {
          isBoundary = true;
          break;
        }
      }
      if (isBoundary) {
        boundary.push(point);
      }
    }

    return boundary;
  }

  /**
   * Analyze contour and classify shape
   */
  private analyzeContour(
    contour: Point[],
    filledArea: number
  ): DetectedShape | null {
    // Calculate bounding box
    const boundingBox = this.calculateBoundingBox(contour);

    console.log(
      `  Analyzing contour: bbox ${boundingBox.width}x${boundingBox.height}, filled area: ${filledArea}`
    );

    // Filter out shapes that are too small
    const minDimension = Math.min(boundingBox.width, boundingBox.height);
    const maxDimension = Math.max(boundingBox.width, boundingBox.height);

    // Very lenient filtering - only reject truly degenerate shapes
    // Allow even very small shapes as long as they have reasonable proportions
    if (minDimension < 3) {
      console.log("  -> Shape too small (min dimension < 3)");
      return null;
    }

    // Filter very thin shapes (likely lines or noise)
    const aspectRatio = maxDimension / minDimension;
    if (aspectRatio > 20) {
      console.log("  -> Shape too elongated (aspect ratio > 20)");
      return null;
    }

    // Calculate center
    const center = this.calculateCentroid(contour);

    // Calculate perimeter from raw contour (for approximation epsilon)
    const rawPerimeter = this.calculatePerimeter(contour);

    // Calculate convex hull on ORIGINAL contour (before approximation)
    // This is crucial for detecting concave shapes like stars
    const convexHull = this.convexHull(contour);
    const hullPerimeter = this.calculatePerimeter(convexHull);
    const convexHullArea = this.calculateArea(convexHull);

    // Calculate solidity early to detect potentially concave shapes (stars)
    const earlySolidity =
      convexHullArea > 0 ? filledArea / convexHullArea : 1.0;
    const isPotentiallyConcave = earlySolidity < 0.85;

    // Approximate polygon (pass concavity hint for better star detection)
    const approxPoly = this.approximatePolygon(
      contour,
      rawPerimeter,
      isPotentiallyConcave
    );

    // Use filled area from flood fill - it's accurate!
    // Shoelace formula on approximated polygons gives wrong areas because vertices skip boundary portions
    const area = filledArea;

    console.log(
      `  Raw perimeter: ${rawPerimeter.toFixed(
        0
      )}, Hull perimeter: ${hullPerimeter.toFixed(0)}, Vertices: ${
        approxPoly.length
      }`
    );
    console.log(`  Area: ${area} pixels`);

    // Classify shape
    const classification = this.classifyShape(
      approxPoly,
      area,
      hullPerimeter,
      convexHull,
      boundingBox
    );

    if (!classification) {
      return null;
    }

    return {
      type: classification.type,
      confidence: classification.confidence,
      boundingBox,
      center,
      area,
    };
  }

  /**
   * Calculate bounding box of contour
   */
  private calculateBoundingBox(contour: Point[]): {
    x: number;
    y: number;
    width: number;
    height: number;
  } {
    let minX = Infinity,
      minY = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity;

    for (const point of contour) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  /**
   * Calculate centroid (center) of contour
   */
  private calculateCentroid(contour: Point[]): Point {
    let sumX = 0,
      sumY = 0;
    for (const point of contour) {
      sumX += point.x;
      sumY += point.y;
    }
    return {
      x: sumX / contour.length,
      y: sumY / contour.length,
    };
  }

  /**
   * Calculate area using shoelace formula
   */
  private calculateArea(contour: Point[]): number {
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
      const j = (i + 1) % contour.length;
      area += contour[i].x * contour[j].y;
      area -= contour[j].x * contour[i].y;
    }
    return Math.abs(area / 2);
  }

  /**
   * Calculate perimeter
   */
  private calculatePerimeter(contour: Point[]): number {
    let perimeter = 0;
    for (let i = 0; i < contour.length; i++) {
      const j = (i + 1) % contour.length;
      const dx = contour[j].x - contour[i].x;
      const dy = contour[j].y - contour[i].y;
      perimeter += Math.sqrt(dx * dx + dy * dy);
    }
    return perimeter;
  }

  /**
   * Approximate polygon using Douglas-Peucker algorithm
   */
  private approximatePolygon(
    contour: Point[],
    perimeter: number,
    isPotentiallyConcave: boolean = false
  ): Point[] {
    if (contour.length < 3) return contour;

    // Adaptive epsilon strategy: try multiple values and pick the best one
    // We want to find the simplest approximation that captures the shape's true geometry

    // For potentially concave shapes (stars), use MUCH less aggressive epsilon
    // to preserve the star points
    const baseEpsilon = isPotentiallyConcave
      ? Math.max(2.0, perimeter * 0.008) // Less aggressive for stars
      : Math.max(3.0, perimeter * 0.02); // Original aggressive for simple shapes

    // For concave shapes, try less aggressive multipliers to preserve geometry
    const epsilonMultipliers = isPotentiallyConcave
      ? [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] // More conservative range
      : [
          0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
          15.0,
        ]; // Original aggressive

    let bestApprox = contour;
    let bestScore = -Infinity;

    for (const mult of epsilonMultipliers) {
      const epsilon = baseEpsilon * mult;
      const approx = this.douglasPeuckerClosed(contour, epsilon);

      if (approx.length < 3) continue;

      // Score based on vertex count
      let score = 0;
      const n = approx.length;

      // For potentially concave shapes (stars), heavily favor 8-14 vertices
      if (isPotentiallyConcave) {
        if (n >= 8 && n <= 12)
          score = 250; // Star sweet spot (10 vertices for 5-point star)
        else if (n >= 6 && n <= 7) score = 180; // Close to star
        else if (n >= 13 && n <= 16) score = 150; // Still possibly star
        else if (n === 5) score = 120; // Could be pentagon
        else if (n === 4) score = 80; // Over-simplified
        else if (n === 3) score = 60; // Very over-simplified
        else if (n >= 17 && n <= 25)
          score = 100 - (n - 17) * 5; // Too many vertices
        else score = 30; // Way off

        // Penalty for over-simplification of concave shapes
        if (n <= 5 && mult >= 3.0) score -= 80;
        if (n <= 4 && mult >= 2.0) score -= 60;
      } else {
        // Original scoring for simple convex shapes
        if (n === 3) score = 200; // Triangle - absolute highest priority
        else if (n === 4)
          score = 200; // Rectangle/Square - absolute highest priority
        else if (n === 5) score = 200; // Pentagon - absolute highest priority
        else if (n === 6) score = 70; // Could be pentagon+1
        else if (n === 7) score = 60; // Could be pentagon+2
        else if (n >= 8 && n <= 10) score = 55 - (n - 8) * 3;
        // Star vertices (shouldn't reach here if concave)
        else if (n >= 11 && n <= 18)
          score = 45 - (n - 11) * 2; // Circle approximation
        else if (n >= 19 && n <= 30)
          score = 30 - (n - 19); // High vertex circle
        else if (n > 30) score = 15 - (n - 30); // Too many vertices

        // HUGE rewards for higher epsilon values that give us exact basic shape counts
        const isComplex = contour.length > 350;
        if (n <= 5) {
          const bonus1 = mult >= 2.0 ? 40 : 0;
          const bonus2 = mult >= 3.0 ? 30 : 0;
          const bonus3 = mult >= 4.0 ? 20 : 0;
          const bonus4 =
            mult >= 8.0 && !isComplex
              ? 30
              : mult >= 8.0 && isComplex && n === 4
              ? 5
              : mult >= 8.0 && n === 5
              ? 35
              : 0;
          const bonus5 =
            mult >= 12.0 && !isComplex
              ? 20
              : mult >= 12.0 && isComplex && n === 4
              ? 0
              : mult >= 12.0 && n === 5
              ? 25
              : 0;
          score += bonus1 + bonus2 + bonus3 + bonus4 + bonus5;
        }

        // Boost scores for 5+ vertices on complex contours (pentagons)
        if (isComplex && n >= 5 && n <= 14) {
          score += 60;
        }

        // Extra boost for 5-6 vertices on very complex contours (likely pentagon)
        if (contour.length > 450 && n >= 5 && n <= 6) {
          score += 40;
        }

        // Heavy penalty for under-simplification
        if (n > 15 && mult < 2.0) score -= 30;
        if (n > 10 && mult < 1.0) score -= 20;

        // Only penalize extreme over-simplification
        if (mult > 15.0 && n < 3) score -= 30;
        else if (mult > 10.0 && n < 3) score -= 20;
      }

      if (score > bestScore) {
        bestScore = score;
        bestApprox = approx;
      }
    }

    console.log(
      `  Approximation: ${contour.length} -> ${
        bestApprox.length
      } vertices (score: ${bestScore.toFixed(0)})`
    );
    return bestApprox;
  }

  /**
   * Douglas-Peucker algorithm for closed polygon approximation
   */
  private douglasPeuckerClosed(points: Point[], epsilon: number): Point[] {
    if (points.length < 3) return points;
    if (points.length > 1000) {
      // For very large contours, subsample first
      const step = Math.floor(points.length / 500);
      const subsampled: Point[] = [];
      for (let i = 0; i < points.length; i += step) {
        subsampled.push(points[i]);
      }
      points = subsampled;
    }

    // Find the point farthest from the first point to split the polygon
    let maxDist = 0;
    let splitIndex = Math.floor(points.length / 2);
    for (let i = 1; i < points.length; i++) {
      const dist = Math.sqrt(
        Math.pow(points[i].x - points[0].x, 2) +
          Math.pow(points[i].y - points[0].y, 2)
      );
      if (dist > maxDist) {
        maxDist = dist;
        splitIndex = i;
      }
    }

    // Split into two parts and simplify each
    const firstHalf = points.slice(0, splitIndex + 1);
    const secondHalf = [...points.slice(splitIndex), points[0]];

    const simplified1 = this.douglasPeucker(firstHalf, epsilon);
    const simplified2 = this.douglasPeucker(secondHalf, epsilon);

    // Combine results, removing duplicates at join points
    const result = [...simplified1.slice(0, -1), ...simplified2.slice(0, -1)];

    // Make sure we have at least 3 points
    if (result.length < 3 && points.length >= 3) {
      // If over-simplified, try with smaller epsilon
      return this.douglasPeuckerClosed(points, epsilon * 0.5);
    }

    return result;
  }

  /**
   * Douglas-Peucker algorithm for polygon approximation
   */
  private douglasPeucker(points: Point[], epsilon: number): Point[] {
    if (points.length < 3) return points;

    // Find point with maximum distance from line segment
    let maxDist = 0;
    let maxIndex = 0;
    const start = points[0];
    const end = points[points.length - 1];

    for (let i = 1; i < points.length - 1; i++) {
      const dist = this.perpendicularDistance(points[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (maxDist > epsilon) {
      const left = this.douglasPeucker(points.slice(0, maxIndex + 1), epsilon);
      const right = this.douglasPeucker(points.slice(maxIndex), epsilon);
      return [...left.slice(0, -1), ...right];
    } else {
      return [start, end];
    }
  }

  /**
   * Calculate perpendicular distance from point to line
   */
  private perpendicularDistance(
    point: Point,
    lineStart: Point,
    lineEnd: Point
  ): number {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;

    if (dx === 0 && dy === 0) {
      return Math.sqrt(
        Math.pow(point.x - lineStart.x, 2) + Math.pow(point.y - lineStart.y, 2)
      );
    }

    const num = Math.abs(
      dy * point.x -
        dx * point.y +
        lineEnd.x * lineStart.y -
        lineEnd.y * lineStart.x
    );
    const den = Math.sqrt(dx * dx + dy * dy);

    return num / den;
  }

  /**
   * Classify shape based on geometric features
   */
  private classifyShape(
    approxPoly: Point[],
    area: number,
    perimeter: number,
    convexHull: Point[],
    boundingBox: { x: number; y: number; width: number; height: number }
  ): { type: DetectedShape["type"]; confidence: number } | null {
    const numVertices = approxPoly.length;

    console.log(
      `Classifying: ${numVertices} vertices, area: ${area.toFixed(
        0
      )}, perimeter: ${perimeter.toFixed(0)}`
    );

    // No minimum area filter here - already filtered in analyzeContour

    // Calculate circularity: 4π * area / perimeter²
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

    // Calculate solidity (ratio of shape area to convex hull area)
    // Solidity should always be <= 1.0 (shape area / convex hull area)
    const convexHullArea = this.calculateArea(convexHull);
    const solidity =
      convexHullArea > 0 ? Math.min(1.0, area / convexHullArea) : 1;

    // Calculate aspect ratio for rectangle detection
    const aspectRatio = boundingBox.width / boundingBox.height;

    console.log(
      `  Circularity: ${circularity.toFixed(3)}, Solidity: ${solidity.toFixed(
        3
      )}, Aspect: ${aspectRatio.toFixed(2)}`
    );

    // IMPORTANT: Check specific vertex counts FIRST before circles
    // This prevents triangles/rectangles from being misclassified as circles

    // PRIORITY 1: Exact vertex matches - highest confidence

    // Triangle: 3 vertices
    if (numVertices === 3) {
      console.log(`  -> TRIANGLE detected (3 vertices, confidence: 0.95)`);
      return { type: "triangle", confidence: 0.95 };
    }

    // 4 vertices: Could be triangle, pentagon, OR rectangle
    if (numVertices === 4) {
      // Use fill ratio and circularity to distinguish
      const boundingBoxArea = boundingBox.width * boundingBox.height;
      const fillRatio = area / boundingBoxArea;

      console.log(
        `    4-vertex analysis: fillRatio=${fillRatio.toFixed(
          2
        )}, circ=${circularity.toFixed(2)}`
      );

      // Triangle: low fill ratio (< 0.65)
      if (fillRatio < 0.65) {
        console.log(
          `  -> TRIANGLE detected (4 vertices, low fill ratio ${fillRatio.toFixed(
            2
          )}, confidence: 0.90)`
        );
        return { type: "triangle", confidence: 0.9 };
      }

      // Pentagon: moderate fill ratio (0.65-0.85) + pentagon-like circularity (0.80-0.95)
      // Pentagons have circularity around 0.85-0.90, NOT > 1.0
      if (
        fillRatio >= 0.65 &&
        fillRatio < 0.85 &&
        circularity >= 0.8 &&
        circularity < 0.95 &&
        aspectRatio < 1.15
      ) {
        console.log(
          `  -> PENTAGON detected (4 vertices, moderate fill + pentagon circ, confidence: 0.85)`
        );
        return { type: "pentagon", confidence: 0.85 };
      }

      // Rectangle: high fill ratio (>= 0.85)
      if (fillRatio >= 0.85) {
        console.log(
          `  -> RECTANGLE detected (4 vertices, high fill ratio ${fillRatio.toFixed(
            2
          )}, confidence: 0.95)`
        );
        return { type: "rectangle", confidence: 0.95 };
      }

      // Default to rectangle if nothing else matches
      console.log(
        `  -> RECTANGLE detected (4 vertices, default, confidence: 0.85)`
      );
      return { type: "rectangle", confidence: 0.85 };
    }

    // Pentagon: 5 vertices
    if (numVertices === 5) {
      console.log(`  -> PENTAGON detected (5 vertices, confidence: 0.95)`);
      return { type: "pentagon", confidence: 0.95 };
    }

    // PRIORITY 2: Near-exact matches with geometric validation

    // Triangle with 6 vertices - check if it's really a triangle
    if (numVertices === 6) {
      const boundingBoxArea = boundingBox.width * boundingBox.height;
      const fillRatio = area / boundingBoxArea;

      // Triangles have low bounding box fill ratio AND low circularity
      // Rectangles can also have low fill ratio if rotated, but have higher circularity
      if (fillRatio < 0.65 && solidity > 0.85) {
        // If circularity is higher (>0.75) and aspect ratio is close to 1, likely a rotated rectangle
        if (circularity > 0.75 && aspectRatio < 1.3) {
          console.log(
            `  -> RECTANGLE detected (6 vertices, rotated rect with low fill ${fillRatio.toFixed(
              2
            )}, confidence: 0.78)`
          );
          return { type: "rectangle", confidence: 0.78 };
        }
        // Otherwise it's a triangle
        console.log(
          `  -> TRIANGLE detected (6 vertices, low fill ratio, confidence: 0.85)`
        );
        return { type: "triangle", confidence: 0.85 };
      }

      // Rectangles fill their bounding box well OR have moderate fill with low circularity
      if (fillRatio > 0.65 && aspectRatio < 5 && solidity > 0.85) {
        // High fill ratio - clearly a rectangle
        if (fillRatio > 0.85) {
          const angles = this.calculatePolygonAngles(approxPoly);
          const rightAngles = angles.filter(
            (a) => Math.abs(a - 90) < 20
          ).length;
          if (rightAngles >= 4) {
            console.log(
              `  -> RECTANGLE detected (6 vertices, high fill + right angles, confidence: 0.82)`
            );
            return { type: "rectangle", confidence: 0.82 };
          }
        }

        // Moderate fill ratio - check circularity to distinguish rectangle from pentagon
        // Rectangles have lower circularity (< 0.85) than pentagons
        if (fillRatio >= 0.65 && fillRatio <= 0.85 && circularity < 0.85) {
          console.log(
            `  -> RECTANGLE detected (6 vertices, moderate fill + low circ ${circularity.toFixed(
              2
            )}, confidence: 0.80)`
          );
          return { type: "rectangle", confidence: 0.8 };
        }
      }

      // Pentagon-like circularity (higher than rectangle)
      if (
        circularity >= 0.85 &&
        circularity < 0.9 &&
        solidity > 0.85 &&
        fillRatio > 0.65
      ) {
        console.log(
          `  -> PENTAGON detected (6 vertices, pentagon-like metrics, confidence: 0.80)`
        );
        return { type: "pentagon", confidence: 0.8 };
      }
    }

    // PRIORITY 3: Higher vertex counts - stars and circles

    // Star detection: concave polygon with low solidity
    if (solidity < 0.85 && numVertices >= 7) {
      // Stars have distinctive low solidity - they're concave
      if (numVertices >= 7 && numVertices <= 20) {
        const confidence =
          solidity < 0.65 ? 0.88 : solidity < 0.8 ? 0.82 : 0.75;
        console.log(
          `  -> STAR detected (${numVertices} vertices, low solidity ${solidity.toFixed(
            2
          )}, confidence: ${confidence.toFixed(2)})`
        );
        return { type: "star", confidence };
      }
    }

    // 7-8 vertices with high solidity - could be triangle, pentagon or rectangle
    if (numVertices === 7 || numVertices === 8) {
      const boundingBoxArea = boundingBox.width * boundingBox.height;
      const fillRatio = area / boundingBoxArea;

      // Triangle: low fill ratio (triangles don't fill bounding box well)
      if (fillRatio < 0.65) {
        console.log(
          `  -> TRIANGLE detected (${numVertices} vertices, low fill ratio ${fillRatio.toFixed(
            2
          )}, confidence: 0.78)`
        );
        return { type: "triangle", confidence: 0.78 };
      }

      // Rectangle: high fill ratio OR elongated shape
      if (fillRatio > 0.85 || (aspectRatio > 1.4 && fillRatio > 0.75)) {
        // Check for rectangular properties
        if (aspectRatio < 5) {
          const angles = this.calculatePolygonAngles(approxPoly);
          const rightAngles = angles.filter(
            (a) => Math.abs(a - 90) < 20
          ).length;
          if (rightAngles >= 3 || aspectRatio > 1.4) {
            console.log(
              `  -> RECTANGLE detected (${numVertices} vertices, rectangular properties, confidence: 0.78)`
            );
            return { type: "rectangle", confidence: 0.78 };
          }
        }
      }

      // Pentagon: moderate circularity + round shape (not elongated)
      if (
        circularity > 0.65 &&
        circularity < 0.9 &&
        solidity > 0.85 &&
        aspectRatio < 1.25
      ) {
        console.log(
          `  -> PENTAGON detected (${numVertices} vertices, pentagon-like metrics, confidence: 0.75)`
        );
        return { type: "pentagon", confidence: 0.75 };
      }
    }

    // Circle detection: high vertex count + high circularity + round shape
    if (numVertices >= 9) {
      // Very high vertex count - but MUST have circular properties AND round shape
      // Aspect ratio should be close to 1.0 (circles are not elongated)
      if (
        numVertices >= 18 &&
        circularity > 0.85 &&
        aspectRatio >= 0.7 &&
        aspectRatio <= 1.4
      ) {
        console.log(
          `  -> CIRCLE detected (${numVertices} vertices, high circ, round, confidence: 0.92)`
        );
        return { type: "circle", confidence: 0.92 };
      }

      // Medium-high vertex count with good metrics
      // Avoid pentagon circularity range (0.90-1.35)
      if (
        numVertices >= 12 &&
        numVertices <= 17 &&
        circularity > 1.35 &&
        solidity > 0.88
      ) {
        const confidence = Math.min(0.9, 0.75 + circularity * 0.15);
        console.log(
          `  -> CIRCLE detected (${numVertices} vertices, very high circ, confidence: ${confidence.toFixed(
            2
          )})`
        );
        return { type: "circle", confidence };
      }

      // Medium vertex count in pentagon circularity range -> pentagon
      // Pentagons typically have circularity between 0.85-1.35
      if (
        numVertices >= 12 &&
        numVertices <= 17 &&
        circularity >= 0.85 &&
        circularity <= 1.35 &&
        solidity > 0.88
      ) {
        console.log(
          `  -> PENTAGON detected (${numVertices} vertices, pentagon circ ${circularity.toFixed(
            2
          )}, confidence: 0.80)`
        );
        return { type: "pentagon", confidence: 0.8 };
      }

      // 9-11 vertices - only if very circular
      if (
        numVertices >= 9 &&
        numVertices <= 11 &&
        circularity > 0.9 &&
        solidity > 0.9
      ) {
        console.log(
          `  -> CIRCLE detected (${numVertices} vertices, very high metrics, confidence: 0.85)`
        );
        return { type: "circle", confidence: 0.85 };
      }
    }

    // FALLBACK: Use geometric properties when vertex count is ambiguous
    // This handles cases where approximation doesn't give exact counts
    console.log(`  -> Trying geometric fallback...`);

    const boundingBoxArea = boundingBox.width * boundingBox.height;
    const fillRatio = area / boundingBoxArea;

    // Rectangle: check first (before triangle) if rectangular aspect ratio
    // Rectangles can have moderate fill ratios due to approximation errors
    if (
      fillRatio > 0.55 &&
      solidity > 0.9 &&
      aspectRatio > 1.3 &&
      aspectRatio < 5
    ) {
      // Non-square rectangles with elongated shape
      console.log(
        `  -> RECTANGLE detected (geometric: elongated ${aspectRatio.toFixed(
          2
        )}, fill ${fillRatio.toFixed(2)}, confidence: 0.78)`
      );
      return { type: "rectangle", confidence: 0.78 };
    }

    // Rectangle: high fill ratio (rectangles/squares fill bounding box)
    if (fillRatio > 0.85 && solidity > 0.9) {
      // Higher confidence for very rectangular shapes
      let confidence = 0.75;
      if (fillRatio > 0.95 && solidity > 0.95 && circularity < 0.9) {
        confidence = 0.88; // Very confident - near-perfect rectangle
      } else if (fillRatio > 0.9 && solidity > 0.92) {
        confidence = 0.82; // Good rectangle characteristics
      }
      console.log(
        `  -> RECTANGLE detected (geometric: high fill ratio ${fillRatio.toFixed(
          2
        )}, confidence: ${confidence.toFixed(2)})`
      );
      return { type: "rectangle", confidence };
    }

    // Triangle: low fill ratio (triangles don't fill their bounding box)
    if (fillRatio < 0.65 && solidity > 0.9) {
      console.log(
        `  -> TRIANGLE detected (geometric: low fill ratio ${fillRatio.toFixed(
          2
        )}, confidence: 0.75)`
      );
      return { type: "triangle", confidence: 0.75 };
    }

    // Pentagon: high circularity but not too high (between rectangle and circle)
    // Pentagon circularity is typically 0.85-1.35
    if (
      circularity >= 0.85 &&
      circularity <= 1.35 &&
      solidity > 0.9 &&
      fillRatio > 0.65 &&
      fillRatio <= 0.85
    ) {
      console.log(
        `  -> PENTAGON detected (geometric: pentagon-like circularity ${circularity.toFixed(
          2
        )}, confidence: 0.72)`
      );
      return { type: "pentagon", confidence: 0.72 };
    }

    // Circle: very high circularity (> 1.35 means very circular, almost perfect)
    if (circularity > 1.05 && solidity > 0.9) {
      console.log(
        `  -> CIRCLE detected (geometric: very high circularity ${circularity.toFixed(
          2
        )}, confidence: 0.80)`
      );
      return { type: "circle", confidence: 0.8 };
    }

    console.log(
      `  -> No match found (vertices: ${numVertices}, circ: ${circularity.toFixed(
        2
      )}, solid: ${solidity.toFixed(2)}, fill: ${fillRatio.toFixed(2)})`
    );
    return null;
  }

  /**
   * Calculate interior angles of a polygon
   */
  private calculatePolygonAngles(points: Point[]): number[] {
    const angles: number[] = [];
    const n = points.length;

    for (let i = 0; i < n; i++) {
      const p1 = points[(i - 1 + n) % n];
      const p2 = points[i];
      const p3 = points[(i + 1) % n];

      // Vectors from current point to neighbors
      const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
      const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };

      // Calculate magnitudes
      const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
      const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);

      if (mag1 > 0.001 && mag2 > 0.001) {
        // Normalize vectors
        const n1 = { x: v1.x / mag1, y: v1.y / mag1 };
        const n2 = { x: v2.x / mag2, y: v2.y / mag2 };

        // Calculate angle using atan2 for better accuracy
        const angle1 = Math.atan2(n1.y, n1.x);
        const angle2 = Math.atan2(n2.y, n2.x);

        // Calculate the angle difference
        let angleDiff = angle2 - angle1;

        // Normalize to [0, 2π]
        while (angleDiff < 0) angleDiff += 2 * Math.PI;
        while (angleDiff > 2 * Math.PI) angleDiff -= 2 * Math.PI;

        // Convert to degrees
        let angle = angleDiff * (180 / Math.PI);

        // We want interior angles, so if > 180, take complement
        if (angle > 180) {
          angle = 360 - angle;
        }

        angles.push(angle);
      } else {
        // Points are too close, skip this angle
        angles.push(90); // Default assumption
      }
    }

    return angles;
  }

  /**
   * Graham scan algorithm for convex hull
   */
  private convexHull(points: Point[]): Point[] {
    if (points.length < 3) return points;

    // Find bottom-most point (or left most in case of tie)
    let bottom = points[0];
    for (const point of points) {
      if (point.y > bottom.y || (point.y === bottom.y && point.x < bottom.x)) {
        bottom = point;
      }
    }

    // Sort points by polar angle with respect to bottom point
    const sorted = [...points].sort((a, b) => {
      const angleA = Math.atan2(a.y - bottom.y, a.x - bottom.x);
      const angleB = Math.atan2(b.y - bottom.y, b.x - bottom.x);
      if (angleA !== angleB) return angleA - angleB;

      // If angles are equal, sort by distance
      const distA = Math.pow(a.x - bottom.x, 2) + Math.pow(a.y - bottom.y, 2);
      const distB = Math.pow(b.x - bottom.x, 2) + Math.pow(b.y - bottom.y, 2);
      return distA - distB;
    });

    const hull: Point[] = [];
    for (const point of sorted) {
      while (
        hull.length >= 2 &&
        this.crossProduct(
          hull[hull.length - 2],
          hull[hull.length - 1],
          point
        ) <= 0
      ) {
        hull.pop();
      }
      hull.push(point);
    }

    return hull;
  }

  /**
   * Calculate cross product for convex hull
   */
  private crossProduct(o: Point, a: Point, b: Point): number {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
