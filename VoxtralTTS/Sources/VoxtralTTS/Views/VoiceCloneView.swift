import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct VoiceCloneView: View {
    let modelPath: URL
    let onVoiceCreated: (String) -> Void

    @StateObject private var recorder = AudioRecorder()
    @State private var voiceName = ""
    @State private var state: CloneState = .idle
    @State private var errorMessage: String?
    @State private var successMessage: String?
    @State private var processedAudioData: Data?
    @State private var previewBuffer: AVAudioPCMBuffer?
    @State private var showingFilePicker = false
    @State private var audioDuration: TimeInterval = 0

    @StateObject private var previewPlayer = AudioPlayer(sampleRate: 24000)

    enum CloneState {
        case idle
        case recording
        case processing
        case previewing
        case saving
        case done
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection

                if state == .done, let success = successMessage {
                    // Success state — show message and reset button
                    VStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 48))
                            .foregroundStyle(.green)
                        Text(success)
                            .font(.headline)
                        Text("Switch to the Generate tab to use this voice.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Button("Clone Another Voice") {
                            resetState()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.vertical, 20)
                } else {
                    recordSection
                    importSection

                    if state == .previewing || processedAudioData != nil {
                        previewSection
                    }

                    if processedAudioData != nil {
                        nameSection
                        saveButton
                    }
                }

                if let error = errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }
            }
            .padding()
        }
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.audio, .wav, .mp3, .aiff,
                                  UTType(filenameExtension: "m4a") ?? .audio],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result)
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.wave.2")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Record or import 3-30 seconds of clear speech to clone a voice.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Record

    private var recordSection: some View {
        VStack(spacing: 12) {
            // Level meter
            if recorder.isRecording {
                levelMeter
                Text(String(format: "%.1fs / 30s", recorder.recordingDuration))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 16) {
                Button(action: toggleRecording) {
                    HStack {
                        Image(systemName: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.title2)
                        Text(recorder.isRecording ? "Stop Recording" : "Record Voice")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(recorder.isRecording ? .red : .accentColor)
                .disabled(state == .processing || state == .saving)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(.background.secondary))
    }

    private var levelMeter: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.secondary.opacity(0.2))
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.green)
                    .frame(width: geometry.size.width * CGFloat(recorder.audioLevel))
                    .animation(.easeOut(duration: 0.1), value: recorder.audioLevel)
            }
        }
        .frame(height: 8)
    }

    // MARK: - Import

    private var importSection: some View {
        Button(action: { showingFilePicker = true }) {
            HStack {
                Image(systemName: "doc.badge.plus")
                Text("Import Audio File")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .disabled(recorder.isRecording || state == .processing || state == .saving)
    }

    // MARK: - Preview

    private var previewSection: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Reference Audio")
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(String(format: "%.1fs", audioDuration))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 16) {
                Button(action: { previewPlayer.togglePlayback() }) {
                    Image(systemName: previewPlayer.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                }

                // Simple progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Color.secondary.opacity(0.2))
                            .frame(height: 4)
                        Rectangle()
                            .fill(Color.accentColor)
                            .frame(width: geometry.size.width * (previewPlayer.duration > 0 ? previewPlayer.currentTime / previewPlayer.duration : 0), height: 4)
                    }
                    .clipShape(Capsule())
                }
                .frame(height: 4)

                Button(action: clearAudio) {
                    Image(systemName: "xmark.circle")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(.background.secondary))
    }

    // MARK: - Name & Save

    private var nameSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Voice Name")
                .font(.subheadline.weight(.medium))
            TextField("e.g., my_voice", text: $voiceName)
                .textFieldStyle(.roundedBorder)
                #if os(iOS)
                .textInputAutocapitalization(.never)
                #endif
                .autocorrectionDisabled()
        }
    }

    private var saveButton: some View {
        Button(action: saveVoice) {
            HStack {
                if state == .saving {
                    ProgressView()
                        .controlSize(.small)
                    Text("Saving...")
                } else {
                    Image(systemName: "checkmark.circle")
                    Text("Create Voice")
                }
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(voiceName.trimmingCharacters(in: .whitespaces).isEmpty || state == .saving)
    }

    // MARK: - Actions

    private func toggleRecording() {
        if recorder.isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        errorMessage = nil
        successMessage = nil
        processedAudioData = nil
        previewBuffer = nil

        Task {
            let granted = await recorder.requestPermission()
            guard granted else {
                errorMessage = "Microphone permission is required."
                return
            }

            do {
                try recorder.startRecording()
                state = .recording
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func stopRecording() {
        state = .processing
        do {
            let buffer = try recorder.stopRecording()
            let wavData = try AudioPreprocessor.process(buffer: buffer)
            processedAudioData = wavData
            loadPreview(from: wavData)
            state = .previewing
        } catch {
            errorMessage = error.localizedDescription
            state = .idle
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            state = .processing
            errorMessage = nil
            successMessage = nil
            processedAudioData = nil

            let accessing = url.startAccessingSecurityScopedResource()
            defer {
                if accessing { url.stopAccessingSecurityScopedResource() }
            }

            do {
                let wavData = try AudioPreprocessor.process(fileURL: url)
                processedAudioData = wavData
                loadPreview(from: wavData)
                state = .previewing

                // Auto-fill name from filename if empty
                if voiceName.isEmpty {
                    voiceName = url.deletingPathExtension().lastPathComponent
                        .replacingOccurrences(of: " ", with: "_")
                        .lowercased()
                }
            } catch {
                errorMessage = error.localizedDescription
                state = .idle
            }

        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }

    private func loadPreview(from wavData: Data) {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voice_preview.wav")
        try? wavData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        guard let file = try? AVAudioFile(forReading: tempURL),
              let format = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: file.fileFormat.sampleRate,
                  channels: 1,
                  interleaved: false
              ),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else { return }

        try? file.read(into: buffer)
        previewPlayer.loadAudio(buffer: buffer)
        audioDuration = previewPlayer.duration
    }

    private func clearAudio() {
        previewPlayer.stop()
        processedAudioData = nil
        previewBuffer = nil
        audioDuration = 0
        state = .idle
    }

    private func saveVoice() {
        guard let wavData = processedAudioData else { return }
        let name = VoiceEmbeddingStore.sanitizedName(voiceName)
        guard !name.isEmpty else {
            errorMessage = "Please enter a voice name."
            return
        }

        if VoiceEmbeddingStore.hasReferenceAudio(name: name, in: modelPath) {
            errorMessage = "A voice with this name already exists."
            return
        }

        state = .saving
        do {
            try VoiceEmbeddingStore.saveCustomVoice(wavData: wavData, name: name, to: modelPath)
            state = .done
            successMessage = "Voice '\(name)' created!"
            onVoiceCreated(name)
        } catch {
            errorMessage = "Failed to save: \(error.localizedDescription)"
            state = .previewing
        }
    }

    private func resetState() {
        state = .idle
        voiceName = ""
        errorMessage = nil
        successMessage = nil
        processedAudioData = nil
        previewBuffer = nil
        audioDuration = 0
        previewPlayer.stop()
    }
}
