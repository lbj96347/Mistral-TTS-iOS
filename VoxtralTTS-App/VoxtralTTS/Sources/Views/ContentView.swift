import SwiftUI
import MLX

struct ContentView: View {
    @StateObject private var viewModel = TTSViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if viewModel.modelLoaded {
                    ttsInterface
                } else if viewModel.isLoadingModel {
                    loadingView
                } else {
                    modelSetupView
                }
            }
            .padding()
            .navigationTitle("Voxtral TTS")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .onAppear {
                viewModel.checkAutoLoadArgument()
            }
        }
    }

    // MARK: - Model Setup

    private var modelSetupView: some View {
        VStack(spacing: 20) {
            Image(systemName: "waveform.circle")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("No Model Loaded")
                .font(.title2)

            Text("Select the Q4 model directory containing config.json and .safetensors files.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }

            Button("Select Model Directory") {
                viewModel.showingDirectoryPicker = true
            }
            .buttonStyle(.borderedProminent)

            if let lastPath = viewModel.lastModelPath {
                Button("Reload Last Model") {
                    viewModel.loadModel(from: URL(fileURLWithPath: lastPath))
                }
                .buttonStyle(.bordered)

                Text(lastPath)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .fileImporter(
            isPresented: $viewModel.showingDirectoryPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    viewModel.loadModel(from: url)
                }
            case .failure(let error):
                viewModel.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Loading View

    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading Model...")
                .font(.headline)
            Text(viewModel.loadingStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - TTS Interface

    private var ttsInterface: some View {
        VStack(spacing: 16) {
            // Voice picker
            HStack {
                Text("Voice:")
                    .font(.subheadline)
                Picker("Voice", selection: $viewModel.selectedVoice) {
                    Text("None").tag("")
                    ForEach(viewModel.availableVoices, id: \.self) { voice in
                        Text(voice).tag(voice)
                    }
                }
                .pickerStyle(.menu)
                Spacer()
            }

            // Text input
            TextEditor(text: $viewModel.inputText)
                .frame(minHeight: 100, maxHeight: 200)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                )
                .overlay(alignment: .topLeading) {
                    if viewModel.inputText.isEmpty {
                        Text("Enter text to synthesize...")
                            .foregroundStyle(.tertiary)
                            .padding(8)
                            .allowsHitTesting(false)
                    }
                }

            // Generate button
            Button(action: { viewModel.generate() }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                        Text("Generating...")
                    } else {
                        Image(systemName: "waveform")
                        Text("Generate Audio")
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.inputText.isEmpty || viewModel.isGenerating)

            // Progress
            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    ProgressView(value: viewModel.generationProgress)
                    Text("Frame \(viewModel.currentFrame) / \(viewModel.maxFrames)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Audio playback
            if viewModel.hasAudio {
                Divider()
                audioPlaybackView
            }

            // Error
            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            // Stats
            if let stats = viewModel.generationStats {
                statsView(stats)
            }

            Spacer()
        }
    }

    // MARK: - Audio Playback

    private var audioPlaybackView: some View {
        VStack(spacing: 8) {
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.secondary.opacity(0.2))
                        .frame(height: 4)
                    Rectangle()
                        .fill(Color.accentColor)
                        .frame(
                            width: geometry.size.width * viewModel.playbackProgress,
                            height: 4
                        )
                }
                .clipShape(Capsule())
            }
            .frame(height: 4)

            // Time labels
            HStack {
                Text(viewModel.audioPlayer.currentTime.formattedTime)
                    .font(.caption.monospacedDigit())
                Spacer()
                Text(viewModel.audioPlayer.duration.formattedTime)
                    .font(.caption.monospacedDigit())
            }
            .foregroundStyle(.secondary)

            // Controls
            HStack(spacing: 24) {
                Button(action: { viewModel.audioPlayer.stop() }) {
                    Image(systemName: "stop.fill")
                        .font(.title3)
                }

                Button(action: { viewModel.audioPlayer.togglePlayback() }) {
                    Image(systemName: viewModel.audioPlayer.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                }

                Button(action: { viewModel.saveAudio() }) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.title3)
                }
            }
            .foregroundStyle(.primary)
        }
    }

    // MARK: - Stats

    private func statsView(_ stats: GenerationStats) -> some View {
        HStack(spacing: 16) {
            statItem("Duration", stats.audioDuration)
            statItem("RTF", String(format: "%.2fx", stats.realTimeFactor))
            statItem("Tokens", "\(stats.tokenCount)")
            statItem("Time", String(format: "%.1fs", stats.processingTime))
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func statItem(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).fontWeight(.medium)
            Text(label).foregroundStyle(.tertiary)
        }
    }
}

// MARK: - Generation Stats

struct GenerationStats {
    let audioDuration: String
    let realTimeFactor: Float
    let tokenCount: Int
    let processingTime: Float
}
