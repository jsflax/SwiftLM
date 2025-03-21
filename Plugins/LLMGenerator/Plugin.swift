import PackagePlugin
import Foundation

@main
struct MyPlugin: CommandPlugin, BuildToolPlugin {
    func createBuildCommands(context: PackagePlugin.PluginContext, target: any PackagePlugin.Target) async throws -> [PackagePlugin.Command] {
        let exportPath = context.package.directoryURL
            .appending(path: "Plugins")
            .appending(path: "LLMGenerator")
            .appending(path: "export.py")
        
        print(exportPath)
        
        return [
//            .buildCommand(displayName: "export",
//                          executable: exportPath,
//                          arguments: <#T##[String]#>, environment: <#T##[String : String]#>, inputFiles: <#T##[URL]#>, outputFiles: <#T##[URL]#>)
        ]
    }
    
    func shell(_ command: String) throws -> String {
        let task = Process()
        let pipe = Pipe()
        
        task.standardOutput = pipe
        task.standardError = pipe
//        task.arguments = ["-c", command]
        task.launchPath = "/usr/bin"
        task.standardInput = nil
        try task.run()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        
        return output
    }
    
    func performCommand(context: PackagePlugin.PluginContext, arguments: [String]) async throws {
        let exportPath = context.package.directoryURL
            .appending(path: "Plugins")
            .appending(path: "LLMGenerator")
            .appending(path: "export.py")
        
        let generatorPath = context.package.directoryURL
            .appending(path: "Plugins")
            .appending(path: "LLMGenerator")
        
        print(exportPath)
        var process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = [generatorPath.appending(path: "run.sh").path(),
                             generatorPath.path()]
        try process.run()
        process.waitUntilExit()
    }
    
    
//    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
//        guard let target = target.sourceModule else { return [] }
//        let inputFiles = target.sourceFiles.filter({ $0.path.extension == "dat" })
//        return try inputFiles.map {
//            let inputFile = $0
//            let inputPath = inputFile.path
//            let outputName = inputPath.stem + ".swift"
//            let outputPath = context.pluginWorkDirectory.appending(outputName)
//            return .buildCommand(
//                displayName: "Generating \(outputName) from \(inputPath.lastComponent)",
//                executable: try context.tool(named: "SomeTool").path,
//                arguments: [ "--verbose", "\(inputPath)", "\(outputPath)" ],
//                inputFiles: [ inputPath, ],
//                outputFiles: [ outputPath ]
//            )
//        }
//    }
}
