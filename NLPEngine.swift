// NLPEngine.swift
// Privacy-Preserving Multi-Task NLP System for iOS
//
// Architecture:
//   1. Apple Natural Language framework → lightweight routing + language detection
//   2. Core ML transformer → deep inference for complex tasks
//   3. Adaptive execution → device-tier-aware model selection
//
// Supported Tasks:
//   - Sentiment Classification
//   - Intent Detection
//   - Emotion Classification
//   - Toxicity Detection
//   - Language Identification
//   - Semantic Similarity

import Foundation
import CoreML
import NaturalLanguage
import Combine

// MARK: - Task Definitions

public enum NLPTask: String, CaseIterable {
    case sentiment = "sentiment"
    case intent    = "intent"
    case emotion   = "emotion"
    case toxicity  = "toxicity"
}

public enum SentimentLabel: String, CaseIterable {
    case negative, neutral, positive
}

public enum IntentLabel: String, CaseIterable {
    case question, command, statement, greeting, complaint
}

public enum EmotionLabel: String, CaseIterable {
    case joy, sadness, anger, fear, surprise, neutral
}

public enum ToxicityLabel: String, CaseIterable {
    case safe, toxic, severeToxic
}

// MARK: - Prediction Result

public struct NLPPrediction {
    public let task: NLPTask
    public let label: String
    public let confidence: Float
    public let allScores: [String: Float]
    public let language: String?
    public let latencyMs: Double
    public let executionPath: ExecutionPath

    public enum ExecutionPath: String {
        case appleNL   = "Apple NL (lightweight)"
        case coreML    = "Core ML Transformer"
        case hybrid    = "Hybrid (NL + CoreML)"
    }
}

// MARK: - Device Capability

public enum DeviceTier: String {
    case high   = "High-End (A17 Pro / M-series)"
    case mid    = "Mid-Range (A15 / A16)"
    case low    = "Budget (A14 and below)"
}

// MARK: - NLP Engine (Main Entry Point)

@MainActor
public final class NLPEngine: ObservableObject {

    // MARK: Properties

    @Published public var isReady: Bool = false
    @Published public var currentTask: String = "Idle"

    private var coreMLModel: MLModel?
    private let tokenizer: BertTokenizer
    private let languageRecognizer = NLLanguageRecognizer()
    private let deviceTier: DeviceTier

    private let maxSequenceLength = 128
    private let queue = DispatchQueue(label: "com.nlpengine.inference", qos: .userInitiated)

    // MARK: Initialization

    public init() {
        self.tokenizer = BertTokenizer()
        self.deviceTier = Self.detectDeviceTier()
        Task { await self.loadModel() }
    }

    // MARK: Model Loading

    private func loadModel() async {
        currentTask = "Loading Core ML model..."

        // Select model variant based on device tier
        let modelName: String
        switch deviceTier {
        case .high:  modelName = "NLPEngine_FP16"    // Full precision
        case .mid:   modelName = "NLPEngine_INT8"    // Quantized
        case .low:   modelName = "NLPEngine_6bit"    // Most compressed
        }

        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            print("⚠️ Core ML model not found: \(modelName). Using Apple NL fallback.")
            isReady = true
            currentTask = "Ready (Apple NL mode)"
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Let Core ML choose CPU/GPU/ANE
            coreMLModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
            isReady = true
            currentTask = "Ready"
            print("✅ Core ML model loaded: \(modelName)")
        } catch {
            print("❌ Model load error: \(error). Falling back to Apple NL.")
            isReady = true
            currentTask = "Ready (Apple NL mode)"
        }
    }

    // MARK: - Public Inference API

    /// Run inference on text for a given task.
    public func predict(text: String, task: NLPTask) async -> NLPPrediction {
        let start = Date()

        // 1. Language detection (Apple NL — always fast)
        let language = detectLanguage(text)

        // 2. Route: use Core ML if available, else Apple NL fallback
        if let model = coreMLModel {
            return await runCoreMLInference(text: text, task: task, language: language, model: model, start: start)
        } else {
            return runAppleNLInference(text: text, task: task, language: language, start: start)
        }
    }

    /// Run semantic similarity between two texts (cosine similarity of embeddings).
    public func similarity(textA: String, textB: String) async -> Float {
        let embedA = await embed(textA)
        let embedB = await embed(textB)
        return cosineSimilarity(embedA, embedB)
    }

    // MARK: - Core ML Inference

    private func runCoreMLInference(
        text: String,
        task: NLPTask,
        language: String?,
        model: MLModel,
        start: Date
    ) async -> NLPPrediction {

        // Tokenize
        let (inputIDs, attentionMask) = tokenizer.encode(text, maxLength: maxSequenceLength)

        // Task ID mapping
        let taskID: Int32
        switch task {
        case .sentiment: taskID = 0
        case .intent:    taskID = 1
        case .emotion:   taskID = 2
        case .toxicity:  taskID = 3
        }

        do {
            // Build feature provider
            let inputIDsArray = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
            let maskArray     = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
            let taskIDArray   = try MLMultiArray(shape: [1], dataType: .int32)

            for i in 0..<maxSequenceLength {
                inputIDsArray[[0, i] as [NSNumber]] = NSNumber(value: inputIDs[i])
                maskArray[[0, i] as [NSNumber]]     = NSNumber(value: attentionMask[i])
            }
            taskIDArray[0] = NSNumber(value: taskID)

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids":      MLFeatureValue(multiArray: inputIDsArray),
                "attention_mask": MLFeatureValue(multiArray: maskArray),
                "task_id":        MLFeatureValue(multiArray: taskIDArray),
            ])

            let prediction = try model.prediction(from: provider)

            // Parse output probabilities
            guard let output = prediction.featureValue(for: "output")?.multiArrayValue else {
                return fallbackPrediction(task: task, language: language, start: start)
            }

            let scores = (0..<output.count).map { Float(truncating: output[$0]) }
            return buildPrediction(
                task: task, scores: scores, language: language,
                executionPath: .coreML, start: start
            )

        } catch {
            print("Core ML inference error: \(error)")
            return runAppleNLInference(text: text, task: task, language: language, start: start)
        }
    }

    // MARK: - Apple NL Fallback Inference

    private func runAppleNLInference(
        text: String,
        task: NLPTask,
        language: String?,
        start: Date
    ) -> NLPPrediction {

        // Use Apple NL sentiment for sentiment task
        if task == .sentiment {
            let tagger = NLTagger(tagSchemes: [.sentimentScore])
            tagger.string = text
            let (sentiment, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
            let score = Float(sentiment?.rawValue ?? "0") ?? 0

            let (label, scores): (String, [String: Float])
            if score > 0.15 {
                label = "positive"
                scores = ["negative": 0.05, "neutral": 0.15, "positive": 0.80]
            } else if score < -0.15 {
                label = "negative"
                scores = ["negative": 0.78, "neutral": 0.17, "positive": 0.05]
            } else {
                label = "neutral"
                scores = ["negative": 0.15, "neutral": 0.70, "positive": 0.15]
            }

            return NLPPrediction(
                task: task,
                label: label,
                confidence: scores[label] ?? 0.5,
                allScores: scores,
                language: language,
                latencyMs: Date().timeIntervalSince(start) * 1000,
                executionPath: .appleNL
            )
        }

        // For other tasks: return heuristic placeholder
        return fallbackPrediction(task: task, language: language, start: start)
    }

    // MARK: - Embeddings

    private func embed(_ text: String) async -> [Float] {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            return Array(repeating: 0, count: 512)
        }
        var vector = [Double](repeating: 0, count: 512)
        embedding.getVector(&vector, for: text)
        return vector.map { Float($0) }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let dot = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (normA * normB)
    }

    // MARK: - Helpers

    private func detectLanguage(_ text: String) -> String? {
        languageRecognizer.reset()
        languageRecognizer.processString(text)
        return languageRecognizer.dominantLanguage?.rawValue
    }

    private static func detectDeviceTier() -> DeviceTier {
        // Real implementation would use sysctlbyname("hw.model") to detect chip
        // Simplified tier detection based on available memory
        let processInfo = ProcessInfo.processInfo
        let memory = processInfo.physicalMemory
        if memory >= 8_000_000_000 { return .high }
        if memory >= 4_000_000_000 { return .mid }
        return .low
    }

    private func buildPrediction(
        task: NLPTask,
        scores: [Float],
        language: String?,
        executionPath: NLPPrediction.ExecutionPath,
        start: Date
    ) -> NLPPrediction {
        let labels = taskLabels(for: task)
        guard !scores.isEmpty else {
            return fallbackPrediction(task: task, language: language, start: start)
        }

        let maxIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) ?? 0
        let label = maxIdx < labels.count ? labels[maxIdx] : "unknown"
        let scoreDict = Dictionary(uniqueKeysWithValues: zip(labels, scores))

        return NLPPrediction(
            task: task,
            label: label,
            confidence: scores[maxIdx],
            allScores: scoreDict,
            language: language,
            latencyMs: Date().timeIntervalSince(start) * 1000,
            executionPath: executionPath
        )
    }

    private func fallbackPrediction(task: NLPTask, language: String?, start: Date) -> NLPPrediction {
        let labels = taskLabels(for: task)
        let uniform = 1.0 / Float(labels.count)
        return NLPPrediction(
            task: task,
            label: labels.first ?? "unknown",
            confidence: uniform,
            allScores: Dictionary(uniqueKeysWithValues: labels.map { ($0, uniform) }),
            language: language,
            latencyMs: Date().timeIntervalSince(start) * 1000,
            executionPath: .appleNL
        )
    }

    private func taskLabels(for task: NLPTask) -> [String] {
        switch task {
        case .sentiment: return ["negative", "neutral", "positive"]
        case .intent:    return ["question", "command", "statement", "greeting", "complaint"]
        case .emotion:   return ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        case .toxicity:  return ["safe", "toxic", "severe_toxic"]
        }
    }
}
