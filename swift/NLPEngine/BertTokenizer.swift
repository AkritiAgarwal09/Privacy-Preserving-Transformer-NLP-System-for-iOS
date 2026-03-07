// BertTokenizer.swift
// Lightweight BERT WordPiece tokenizer for on-device inference.
// Handles: lowercasing, basic tokenization, WordPiece subword splitting.
//
// NOTE: In production, load vocab.txt from the model bundle.
// This implementation uses a simplified vocab subset for demonstration.

import Foundation

public final class BertTokenizer {

    // MARK: - Special Tokens
    static let padToken  = "[PAD]"
    static let unkToken  = "[UNK]"
    static let clsToken  = "[CLS]"
    static let sepToken  = "[SEP]"
    static let maskToken = "[MASK]"

    static let padID: Int32  = 0
    static let unkID: Int32  = 100
    static let clsID: Int32  = 101
    static let sepID: Int32  = 102
    static let maskID: Int32 = 103

    private var vocab: [String: Int32] = [:]
    private var idToToken: [Int32: String] = [:]

    public init() {
        loadVocab()
    }

    // MARK: - Encode

    /// Encode text to (input_ids, attention_mask) arrays of length `maxLength`.
    public func encode(_ text: String, maxLength: Int) -> ([Int32], [Int32]) {
        let tokens = tokenize(text)
        var ids: [Int32] = [Self.clsID]

        for token in tokens {
            let id = vocab[token] ?? Self.unkID
            ids.append(id)
            if ids.count >= maxLength - 1 { break }
        }
        ids.append(Self.sepID)

        // Pad or truncate
        let activeLen = min(ids.count, maxLength)
        var inputIDs    = Array(ids.prefix(activeLen))
        var attentionMask = Array(repeating: Int32(1), count: activeLen)

        while inputIDs.count < maxLength {
            inputIDs.append(Self.padID)
            attentionMask.append(0)
        }

        return (inputIDs, attentionMask)
    }

    // MARK: - Tokenization

    private func tokenize(_ text: String) -> [String] {
        let lowercased = text.lowercased()
        let words = basicTokenize(lowercased)
        var tokens: [String] = []
        for word in words {
            tokens.append(contentsOf: wordPiece(word))
        }
        return tokens
    }

    private func basicTokenize(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""

        for char in text {
            if char.isLetter || char.isNumber {
                current.append(char)
            } else {
                if !current.isEmpty {
                    result.append(current)
                    current = ""
                }
                if !char.isWhitespace {
                    result.append(String(char))
                }
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private func wordPiece(_ word: String) -> [String] {
        guard !word.isEmpty else { return [] }
        if vocab[word] != nil { return [word] }

        var tokens: [String] = []
        var start = word.startIndex
        var isBad = false

        while start < word.endIndex {
            var end = word.endIndex
            var found: String? = nil

            while start < end {
                let sub = String(word[start..<end])
                let candidate = (start == word.startIndex) ? sub : "##" + sub
                if vocab[candidate] != nil {
                    found = candidate
                    break
                }
                end = word.index(before: end)
            }

            if let token = found {
                tokens.append(token)
                start = end
            } else {
                isBad = true
                break
            }
        }

        return isBad ? [Self.unkToken] : tokens
    }

    // MARK: - Vocab Loading

    private func loadVocab() {
        // In production: load from Bundle.main.url(forResource: "vocab", withExtension: "txt")
        // Core 500-token subset for demonstration
        let specialTokens: [(String, Int32)] = [
            ("[PAD]", 0), ("[UNK]", 100), ("[CLS]", 101), ("[SEP]", 102), ("[MASK]", 103)
        ]

        // Common subwords and words that would appear in NLP tasks
        let commonTokens: [String] = [
            "i", "am", "the", "a", "an", "and", "or", "but", "not", "is", "it",
            "this", "that", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "up", "about", "into", "then", "than", "so", "no", "if",
            "love", "hate", "like", "feel", "think", "know", "want", "need",
            "good", "great", "bad", "terrible", "amazing", "awful", "excellent",
            "product", "service", "quality", "price", "experience", "app",
            "happy", "sad", "angry", "scared", "surprised", "excited", "bored",
            "what", "how", "when", "where", "why", "who", "which",
            "please", "thank", "sorry", "help", "stop", "go", "come", "do",
            "hello", "hi", "hey", "good", "morning", "afternoon", "evening",
            "you", "me", "we", "they", "he", "she", "my", "your", "our",
            "very", "really", "so", "too", "more", "most", "much", "many",
            "safe", "toxic", "danger", "harm", "threat", "abuse", "insult",
            "##ing", "##ed", "##er", "##ly", "##s", "##es", "##tion", "##ness",
            "un", "re", "dis", "pre", "over", "under", "out", "up",
        ]

        var currentID: Int32 = 200
        for (token, id) in specialTokens {
            vocab[token] = id
            idToToken[id] = token
        }
        for token in commonTokens {
            if vocab[token] == nil {
                vocab[token] = currentID
                idToToken[currentID] = token
                currentID += 1
            }
        }
    }
}
