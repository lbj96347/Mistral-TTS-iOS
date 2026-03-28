import SwiftUI

@main
struct VoxtralTTSApp: App {

    init() {
        // Check for --test-load CLI flag
        let args = CommandLine.arguments
        if let idx = args.firstIndex(of: "--test-load"), idx + 1 < args.count {
            testLoadModel(path: args[idx + 1])
            Foundation.exit(0)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
