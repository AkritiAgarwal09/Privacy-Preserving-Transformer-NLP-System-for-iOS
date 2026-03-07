// ContentView.swift
// SwiftUI Interface for the Multi-Task NLP Engine
//
// UI Structure:
//   ┌─ NavigationStack ──────────────────────────────────┐
//   │  ┌─ Header ──────────────────────────────────────┐ │
//   │  │  Engine status + device tier badge            │ │
//   │  └───────────────────────────────────────────────┘ │
//   │  ┌─ Task Selector ────────────────────────────────┐ │
//   │  │  Horizontal pill tabs: Sentiment/Intent/...   │ │
//   │  └───────────────────────────────────────────────┘ │
//   │  ┌─ Text Input ──────────────────────────────────┐ │
//   │  │  TextField + Analyze button                   │ │
//   │  └───────────────────────────────────────────────┘ │
//   │  ┌─ Results Panel ────────────────────────────────┐ │
//   │  │  Label + Confidence bar + Score breakdown     │ │
//   │  │  Latency + Execution path badge               │ │
//   │  └───────────────────────────────────────────────┘ │
//   │  ┌─ Benchmark Panel ──────────────────────────────┐ │
//   │  │  Size/Latency/Memory stats cards              │ │
//   │  └───────────────────────────────────────────────┘ │
//   └────────────────────────────────────────────────────┘

import SwiftUI
import Combine

struct ContentView: View {

    @StateObject private var engine = NLPEngine()
    @State private var inputText: String = ""
    @State private var selectedTask: NLPTask = .sentiment
    @State private var prediction: NLPPrediction? = nil
    @State private var isAnalyzing: Bool = false
    @State private var showBenchmarks: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    engineStatusHeader
                    taskSelector
                    inputSection
                    if let pred = prediction {
                        resultPanel(pred)
                    }
                    benchmarkToggle
                    if showBenchmarks {
                        benchmarkPanel
                    }
                }
                .padding()
            }
            .navigationTitle("NLP Engine")
            .navigationBarTitleDisplayMode(.large)
        }
    }

    // MARK: - Engine Status Header

    var engineStatusHeader: some View {
        HStack {
            Circle()
                .fill(engine.isReady ? Color.green : Color.orange)
                .frame(width: 10, height: 10)
                .animation(.easeInOut(duration: 0.5).repeatForever(), value: !engine.isReady)

            Text(engine.isReady ? "Engine Ready" : engine.currentTask)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()

            DeviceTierBadge()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Task Selector

    var taskSelector: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(NLPTask.allCases, id: \.self) { task in
                    TaskPillButton(
                        task: task,
                        isSelected: selectedTask == task
                    ) {
                        withAnimation(.spring(response: 0.3)) {
                            selectedTask = task
                            prediction = nil
                        }
                    }
                }
            }
            .padding(.horizontal, 2)
        }
    }

    // MARK: - Input Section

    var inputSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Enter Text")
                .font(.headline)

            ZStack(alignment: .topLeading) {
                if inputText.isEmpty {
                    Text(placeholderText(for: selectedTask))
                        .foregroundStyle(.tertiary)
                        .padding(.top, 8)
                        .padding(.leading, 4)
                }
                TextEditor(text: $inputText)
                    .frame(minHeight: 100, maxHeight: 160)
                    .scrollContentBackground(.hidden)
            }
            .padding(12)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))

            Button(action: analyze) {
                HStack {
                    if isAnalyzing {
                        ProgressView()
                            .tint(.white)
                            .scaleEffect(0.9)
                    } else {
                        Image(systemName: "brain.filled.head.profile")
                    }
                    Text(isAnalyzing ? "Analyzing..." : "Analyze")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(engine.isReady && !inputText.isEmpty ? Color.accentColor : Color.gray)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(!engine.isReady || inputText.isEmpty || isAnalyzing)
            .animation(.easeInOut(duration: 0.2), value: isAnalyzing)
        }
    }

    // MARK: - Result Panel

    @ViewBuilder
    func resultPanel(_ pred: NLPPrediction) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Result")
                    .font(.headline)
                Spacer()
                ExecutionPathBadge(path: pred.executionPath)
            }

            // Primary label + confidence
            HStack(alignment: .firstTextBaseline) {
                Text(pred.label.capitalized)
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundStyle(labelColor(for: pred))
                Spacer()
                VStack(alignment: .trailing) {
                    Text("\(Int(pred.confidence * 100))%")
                        .font(.title2.bold())
                    Text("confidence")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Confidence bar
            ConfidenceBar(confidence: pred.confidence, color: labelColor(for: pred))

            Divider()

            // All scores breakdown
            VStack(spacing: 8) {
                ForEach(pred.allScores.sorted(by: { $0.value > $1.value }), id: \.key) { label, score in
                    ScoreRow(label: label, score: score, isTop: label == pred.label)
                }
            }

            Divider()

            // Metadata
            HStack {
                MetadataChip(
                    icon: "clock",
                    label: String(format: "%.1f ms", pred.latencyMs)
                )
                if let lang = pred.language {
                    MetadataChip(icon: "globe", label: lang.uppercased())
                }
                MetadataChip(icon: "lock.shield", label: "On-Device")
                Spacer()
            }
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    // MARK: - Benchmark Panel

    var benchmarkToggle: some View {
        Button {
            withAnimation(.spring(response: 0.4)) {
                showBenchmarks.toggle()
            }
        } label: {
            HStack {
                Image(systemName: "chart.bar.xaxis")
                Text("Compression Benchmarks")
                    .fontWeight(.medium)
                Spacer()
                Image(systemName: showBenchmarks ? "chevron.up" : "chevron.down")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        }
        .buttonStyle(.plain)
    }

    var benchmarkPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Core ML Compression Variants")
                .font(.headline)

            ForEach(CompressionVariant.allVariants, id: \.name) { variant in
                CompressionVariantRow(variant: variant)
            }

            Text("* Selected deployment: final_optimized (FP16 + palettized attention)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.top, 4)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .transition(.opacity.combined(with: .move(edge: .top)))
    }

    // MARK: - Actions

    private func analyze() {
        guard !inputText.isEmpty else { return }
        isAnalyzing = true
        prediction = nil

        Task {
            let result = await engine.predict(text: inputText, task: selectedTask)
            await MainActor.run {
                withAnimation(.spring(response: 0.4)) {
                    prediction = result
                    isAnalyzing = false
                }
            }
        }
    }

    // MARK: - Helpers

    private func placeholderText(for task: NLPTask) -> String {
        switch task {
        case .sentiment: return "e.g. This product is absolutely amazing!"
        case .intent:    return "e.g. Can you help me reset my password?"
        case .emotion:   return "e.g. I just got the promotion I worked so hard for!"
        case .toxicity:  return "e.g. Type any text to check for safety..."
        }
    }

    private func labelColor(for pred: NLPPrediction) -> Color {
        switch pred.task {
        case .sentiment:
            switch pred.label {
            case "positive": return .green
            case "negative": return .red
            default: return .orange
            }
        case .toxicity:
            switch pred.label {
            case "safe": return .green
            case "toxic": return .orange
            case "severe_toxic": return .red
            default: return .gray
            }
        case .emotion:
            switch pred.label {
            case "joy": return .yellow
            case "anger": return .red
            case "sadness": return .blue
            case "fear": return .purple
            case "surprise": return .orange
            default: return .gray
            }
        default:
            return .accentColor
        }
    }
}

// MARK: - Sub-Components

struct TaskPillButton: View {
    let task: NLPTask
    let isSelected: Bool
    let action: () -> Void

    var icon: String {
        switch task {
        case .sentiment: return "heart.text.square"
        case .intent:    return "text.bubble"
        case .emotion:   return "face.smiling"
        case .toxicity:  return "shield.lefthalf.filled"
        }
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.caption)
                Text(task.rawValue.capitalized)
                    .font(.subheadline.weight(.medium))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(isSelected ? Color.accentColor : Color(.secondarySystemBackground))
            .foregroundStyle(isSelected ? .white : .primary)
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

struct ConfidenceBar: View {
    let confidence: Float
    let color: Color

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color(.systemGray5))
                Capsule()
                    .fill(color)
                    .frame(width: geo.size.width * CGFloat(confidence))
                    .animation(.spring(response: 0.6, dampingFraction: 0.8), value: confidence)
            }
        }
        .frame(height: 8)
    }
}

struct ScoreRow: View {
    let label: String
    let score: Float
    let isTop: Bool

    var body: some View {
        HStack {
            Text(label.replacingOccurrences(of: "_", with: " ").capitalized)
                .font(.subheadline)
                .fontWeight(isTop ? .semibold : .regular)
                .foregroundStyle(isTop ? .primary : .secondary)
            Spacer()
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(.systemGray5)).frame(height: 6)
                    Capsule()
                        .fill(isTop ? Color.accentColor : Color(.systemGray3))
                        .frame(width: geo.size.width * CGFloat(score), height: 6)
                }
                .frame(height: 6)
            }
            .frame(width: 100, height: 6)
            Text(String(format: "%.0f%%", score * 100))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .trailing)
        }
    }
}

struct MetadataChip: View {
    let icon: String
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
            Text(label)
                .font(.caption)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color(.tertiarySystemBackground))
        .clipShape(Capsule())
    }
}

struct ExecutionPathBadge: View {
    let path: NLPPrediction.ExecutionPath

    var color: Color {
        switch path {
        case .coreML: return .purple
        case .appleNL: return .blue
        case .hybrid: return .teal
        }
    }

    var body: some View {
        Text(path.rawValue)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}

struct DeviceTierBadge: View {
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "cpu")
                .font(.caption2)
            Text("Neural Engine")
                .font(.caption)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(Color.purple.opacity(0.12))
        .foregroundStyle(.purple)
        .clipShape(Capsule())
    }
}

// MARK: - Compression Benchmark Data

struct CompressionVariant {
    let name: String
    let sizeMB: Double
    let latencyMs: Double
    let accuracyRetention: Double
    let isSelected: Bool

    static let allVariants: [CompressionVariant] = [
        .init(name: "FP32 Baseline",     sizeMB: 255.4, latencyMs: 87.3, accuracyRetention: 100.0, isSelected: false),
        .init(name: "FP16 CoreML",       sizeMB: 127.8, latencyMs: 52.1, accuracyRetention: 99.8,  isSelected: false),
        .init(name: "INT8 Quantized",    sizeMB: 68.3,  latencyMs: 41.6, accuracyRetention: 99.1,  isSelected: false),
        .init(name: "6-bit Palettized",  sizeMB: 47.2,  latencyMs: 38.9, accuracyRetention: 98.4,  isSelected: false),
        .init(name: "Pruned + INT8",     sizeMB: 31.6,  latencyMs: 34.2, accuracyRetention: 97.6,  isSelected: false),
        .init(name: "Final Optimized ★", sizeMB: 94.1,  latencyMs: 38.0, accuracyRetention: 98.2,  isSelected: true),
    ]
}

struct CompressionVariantRow: View {
    let variant: CompressionVariant

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(variant.name)
                    .font(.subheadline)
                    .fontWeight(variant.isSelected ? .semibold : .regular)
                Text("\(String(format: "%.1f", variant.sizeMB)) MB")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("\(String(format: "%.1f", variant.latencyMs)) ms")
                    .font(.caption.monospacedDigit())
                Text("\(String(format: "%.1f", variant.accuracyRetention))% acc")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(variant.isSelected ? Color.accentColor.opacity(0.08) : Color(.tertiarySystemBackground))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(variant.isSelected ? Color.accentColor : Color.clear, lineWidth: 1.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

#Preview {
    ContentView()
}
