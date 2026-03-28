import SwiftUI
import MLX

struct ContentView: View {
    @StateObject private var viewModel = TTSViewModel()
    @State private var selectedTab = 0

    var body: some View {
        Group {
            if viewModel.modelLoaded {
                mainTabView
            } else if viewModel.isLoadingModel {
                NavigationStack {
                    loadingView
                        .navigationTitle("Voxtral TTS")
                        #if os(iOS)
                        .navigationBarTitleDisplayMode(.inline)
                        #endif
                }
            } else {
                NavigationStack {
                    modelSetupView
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
        }
        .sheet(isPresented: $viewModel.showingAPIKeySettings) {
            apiKeySettingsSheet
        }
        #if os(iOS)
        .sheet(isPresented: Binding(
            get: { viewModel.shareURL != nil },
            set: { if !$0 { viewModel.shareURL = nil } }
        )) {
            if let url = viewModel.shareURL {
                ActivityView(activityItems: [url]) {
                    try? FileManager.default.removeItem(at: url)
                    viewModel.shareURL = nil
                }
            }
        }
        #endif
    }

    // MARK: - Tab View (after model loaded)

    private var mainTabView: some View {
        TabView(selection: $selectedTab) {
            // Tab 1: Generate
            NavigationStack {
                ScrollView {
                    ttsInterface
                        .padding()
                }
                .navigationTitle("Generate")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        Menu {
                            Button(action: { viewModel.showingAPIKeySettings = true }) {
                                Label("API Key Settings", systemImage: "key")
                            }
                        } label: {
                            Image(systemName: "gearshape")
                        }
                    }
                }
            }
            .tabItem {
                Label("Generate", systemImage: "waveform")
            }
            .tag(0)

            // Tab 2: Clone Voice
            NavigationStack {
                if let modelPath = viewModel.modelPath {
                    VoiceCloneView(
                        modelPath: modelPath,
                        onVoiceCreated: { name in
                            viewModel.onVoiceCreated(name)
                            selectedTab = 0  // Switch to Generate tab
                        }
                    )
                    .navigationTitle("Clone Voice")
                    #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                    #endif
                }
            }
            .tabItem {
                Label("Clone Voice", systemImage: "person.wave.2")
            }
            .tag(1)
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
                    if let url = viewModel.resolveBookmark() {
                        viewModel.loadModel(from: url)
                    } else {
                        viewModel.errorMessage = "Saved bookmark is invalid. Please select the model directory again."
                    }
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

    // MARK: - TTS Interface (Generate tab)

    private var ttsInterface: some View {
        VStack(spacing: 16) {
            // Voice picker
            voicePickerSection

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

            // API indicator for cloned voices
            if viewModel.isCustomVoiceSelected {
                HStack(spacing: 4) {
                    Image(systemName: "cloud")
                        .font(.caption2)
                    Text("This voice uses the Mistral API for generation")
                        .font(.caption)
                }
                .foregroundStyle(.secondary)
            }

            // Generate button
            Button(action: { viewModel.generate() }) {
                HStack {
                    if viewModel.isGenerating {
                        ProgressView()
                            .controlSize(.small)
                        Text(viewModel.isCustomVoiceSelected ? "Generating (API)..." : "Generating...")
                    } else {
                        Image(systemName: "waveform")
                        Text("Generate Audio")
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.inputText.isEmpty || viewModel.isGenerating)

            // Progress + Cancel
            if viewModel.isGenerating {
                VStack(spacing: 4) {
                    if viewModel.isCustomVoiceSelected {
                        ProgressView()
                        HStack {
                            Text("Waiting for API response...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Cancel") {
                                viewModel.cancelGeneration()
                            }
                            .font(.caption)
                            .foregroundStyle(.red)
                        }
                    } else {
                        ProgressView(value: viewModel.generationProgress)
                        HStack {
                            Text(viewModel.totalChunks > 1
                                ? "Chunk \(viewModel.currentChunk + 1)/\(viewModel.totalChunks) — Frame \(viewModel.currentFrame)/\(viewModel.maxFrames)"
                                : "Frame \(viewModel.currentFrame) / \(viewModel.maxFrames)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Cancel") {
                                viewModel.cancelGeneration()
                            }
                            .font(.caption)
                            .foregroundStyle(.red)
                        }
                    }
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
        }
    }

    // MARK: - Voice Picker

    private var voicePickerSection: some View {
        HStack {
            Text("Voice:")
                .font(.subheadline)

            Picker("Voice", selection: $viewModel.selectedVoice) {
                Text("None").tag("")

                if !viewModel.presetVoices.isEmpty {
                    Section("Preset Voices") {
                        ForEach(viewModel.presetVoices, id: \.self) { voice in
                            Text(voice).tag(voice)
                        }
                    }
                }

                if !viewModel.customVoices.isEmpty {
                    Section("Cloned Voices") {
                        ForEach(viewModel.customVoices, id: \.self) { voice in
                            Label(voice, systemImage: "person.wave.2")
                                .tag(voice)
                        }
                    }
                }
            }
            .pickerStyle(.menu)

            Spacer()

            // Delete custom voice
            if !viewModel.selectedVoice.isEmpty,
               viewModel.isCustomVoiceSelected {
                Button(action: { viewModel.deleteCustomVoice(viewModel.selectedVoice) }) {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .foregroundStyle(.red)
                .help("Delete this custom voice")
            }
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

    // MARK: - API Key Settings Sheet

    private var apiKeySettingsSheet: some View {
        NavigationStack {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Mistral API Key")
                        .font(.headline)
                    Text("Required for cloned voice generation. Get a free key (no credit card) at console.mistral.ai.")
                        .font(.callout)
                        .foregroundStyle(.secondary)

                    SecureField("API Key", text: Binding(
                        get: { viewModel.apiKey },
                        set: { viewModel.apiKey = $0 }
                    ))
                    .textFieldStyle(.roundedBorder)
                    #if os(iOS)
                    .textInputAutocapitalization(.never)
                    #endif
                    .autocorrectionDisabled()
                }

                if viewModel.apiKey.isEmpty {
                    Text("Preset voices work without an API key (on-device generation).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        viewModel.showingAPIKeySettings = false
                    }
                }
            }
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
