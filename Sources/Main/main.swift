//import Foundation
//import LlamaANE
////import CoreML
////@preconcurrency import Generation
////@preconcurrency import Tokenizers
////

////
////let config = MLModelConfiguration()
////config.computeUnits = .cpuAndNeuralEngine
////var generationConfig = GenerationConfig(maxLength: 2048,
////                                        maxNewTokens: 64,
////                                        doSample: true,
////                                        temperature: 0.6,
////                                        topK: 50)
//////        let generationConfig = GenerationConfig(maxNewTokens: 128)
////config.optimizationHints.specializationStrategy = .fastPrediction
////config.optimizationHints.reshapeFrequency = .infrequent
////config.allowLowPrecisionAccumulationOnGPU = true // Use FP16 on GPU
////let tools = await MyTools()
////let lm = try LanguageModel(model: Llama_3_2_3B_Instruct_Fixed(configuration: config).model
//////                 ,          tools: tools
////)
////generationConfig.eosTokenId = await lm.tokenizer.eosTokenId//("<|eot_id|>", text: "")
////let encoder = JSONEncoder()
////encoder.outputFormatting = .prettyPrinted
////let encoded = try await encoder.encode(tools.tools.values.map(\.1))
//////for _ in 0..<10 {
//////let stream = try await lm.infer(prompt: """
//////<|begin_of_text|><|start_header_id|>system<|end_header_id|>
//////You are an expert in composing functions. You are given a question and a set of possible functions.
//////Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
//////If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
//////also point it out. You should only return the function call in tools call sections.
//////
//////If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
//////You SHOULD NOT include any other text in the response.
//////
//////Here is a list of functions in JSON format that you can invoke.
//////
//////\(String(data: encoded, encoding: .utf8)!)
//////<|eot_id|><|start_header_id|>user<|end_header_id|>
//////
//////What is today's date?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
//////""", config: generationConfig)
//////Task {
////let stream = try await lm.makeSession().infer(prompt: "How are you today?")
////for await str in stream {
////    print(str, terminator: "")
////}
//////}
////
//////try await Task {
//////    let stream = try await lm.makeSession().infer(prompt: """
//////<|begin_of_text|><|start_header_id|>system<|end_header_id|>
//////You are an AI Assistant. Summarize the emails you are given in 2 sentences or less. Do *NOT* start sentences with "The email is" or other similar phrases.
//////<|eot_id|><|start_header_id|>user<|end_header_id|>
//////*Email*
//////- From: notifications@github.com
//////- Subject: Re: [MailCore/mailcore2] Having issues adding MailCore2 via SPM to
////// iOS project (Issue #2003)
//////- Body: I have filed a pull request to have SPM build from source: #2011. It no longer uses a binary and you should no longer face the issue you are facing. If you wish to use the new Package, please use https://github.com/jsflax/mailcore2/ as your Swift Package dependency: In Xcode click File -> Swift Packages -> Add Package Dependency... Paste the following URL: https://github.com/jsflax/mailcore2 On the Choose Package Options screen, under Rules switch from Version to Branch (Branch: master) will be the default Click Next -> Next -> Finish — Reply to this email directly, view it on GitHub, or unsubscribe. You are receiving this because you are subscribed to this thread.Message ID: <MailCore/mailcore2/issues/2003/2509793069@github.com><|eot_id|><|start_header_id|>assistant<|end_header_id|>
//////""", config: generationConfig)
//////    for await str in stream {
//////        print(str, terminator: "")
//////    }
//////}.value
////sleep(5)
////print("")
////
//@JSONSchema struct Trip {
//    let location: String
//    let daysTraveling: Int
//}
//
//let lm = try LlamaANE.LanguageModel(model: Llama_3_2_3B_Instruct_uncensored_Int4().model
////                                            ,
////                                            temperature: 1.2,
////                                            topK: 50,
////                                            topP: 0.9,
////                                            repetitionPenalty: 1.1
//)
//let session = try await lm.makeSession(Trip.self, systemPrompt: """
//You are the world's greatest itinerary creator. You are to provide the user
//with a simple JSON response that gives them a basic trip itinerary.
//""")
//
//var tokens = ""
//for await token in await session.infer(prompt: "I want to go somewhere cold and be away for 40 days.") {
//    tokens += token
//    print(token)
//}
//print(tokens)
