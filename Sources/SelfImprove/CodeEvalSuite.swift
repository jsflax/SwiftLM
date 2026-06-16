import Foundation

/// Pure-function Swift coding tasks for the code-execution flywheel. TRAIN tasks feed best-of-N
/// generation → compile/test verify → distill; EVAL tasks (disjoint) measure execution-pass@1
/// before/after. The 7B-4bit is NOT saturated here (unlike the 5 trivial MCP tools), so this is
/// the surface where the flywheel can show real LIFT. Each `test` is the HIDDEN grader (top-level
/// checks → print("ALL_PASS") or exit(1)); it is never shown to the model being graded.
public enum CodeEvalSuite {
    private static func task(_ id: String, _ sig: String, _ desc: String, _ test: String) -> CodeTask {
        CodeTask(id: id, signature: sig, desc: desc, test: test)
    }

    public static let trainTasks: [CodeTask] = [
        task("reverseString", "func reverseString(_ s: String) -> String",
             "returns the input string reversed", """
             if reverseString("hello") != "olleh" { print("F1"); exit(1) }
             if reverseString("") != "" { print("F2"); exit(1) }
             if reverseString("ab") != "ba" { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("isPalindrome", "func isPalindrome(_ s: String) -> Bool",
             "returns true iff the string reads the same forwards and backwards (empty string is a palindrome)", """
             if isPalindrome("racecar") != true { print("F1"); exit(1) }
             if isPalindrome("hello") != false { print("F2"); exit(1) }
             if isPalindrome("") != true { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("factorial", "func factorial(_ n: Int) -> Int",
             "returns n! (factorial); factorial(0) is 1", """
             if factorial(0) != 1 { print("F1"); exit(1) }
             if factorial(5) != 120 { print("F2"); exit(1) }
             if factorial(1) != 1 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("gcd", "func gcd(_ a: Int, _ b: Int) -> Int",
             "returns the greatest common divisor of a and b", """
             if gcd(12, 8) != 4 { print("F1"); exit(1) }
             if gcd(17, 5) != 1 { print("F2"); exit(1) }
             if gcd(100, 10) != 10 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("sumArray", "func sumArray(_ a: [Int]) -> Int",
             "returns the sum of all elements (0 for an empty array)", """
             if sumArray([1, 2, 3]) != 6 { print("F1"); exit(1) }
             if sumArray([]) != 0 { print("F2"); exit(1) }
             if sumArray([-1, 1]) != 0 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("countVowels", "func countVowels(_ s: String) -> Int",
             "returns the number of vowels (a, e, i, o, u, both lowercase and uppercase)", """
             if countVowels("hello") != 2 { print("F1"); exit(1) }
             if countVowels("xyz") != 0 { print("F2"); exit(1) }
             if countVowels("AEIOU") != 5 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("fizzbuzz", "func fizzbuzz(_ n: Int) -> [String]",
             "returns an array for 1...n where multiples of 3 are \"Fizz\", of 5 are \"Buzz\", of 15 are \"FizzBuzz\", otherwise the number as a String", """
             if fizzbuzz(5) != ["1", "2", "Fizz", "4", "Buzz"] { print("F1"); exit(1) }
             let r = fizzbuzz(15)
             if r.count != 15 { print("F2"); exit(1) }
             if r.last != "FizzBuzz" { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("isPrime", "func isPrime(_ n: Int) -> Bool",
             "returns true iff n is a prime number (n < 2 is not prime)", """
             if isPrime(2) != true { print("F1"); exit(1) }
             if isPrime(1) != false { print("F2"); exit(1) }
             if isPrime(9) != false { print("F3"); exit(1) }
             if isPrime(13) != true { print("F4"); exit(1) }
             print("ALL_PASS")
             """),
        task("maxOf", "func maxOf(_ a: [Int]) -> Int",
             "returns the maximum element (the array is non-empty)", """
             if maxOf([3, 1, 2]) != 3 { print("F1"); exit(1) }
             if maxOf([5]) != 5 { print("F2"); exit(1) }
             if maxOf([-1, -5, -2]) != -1 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("fibonacci", "func fibonacci(_ n: Int) -> Int",
             "returns the nth Fibonacci number where fibonacci(0) is 0 and fibonacci(1) is 1", """
             if fibonacci(0) != 0 { print("F1"); exit(1) }
             if fibonacci(1) != 1 { print("F2"); exit(1) }
             if fibonacci(7) != 13 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
    ]

    public static let evalTasks: [CodeTask] = [
        task("celsiusToFahrenheit", "func celsiusToFahrenheit(_ c: Double) -> Double",
             "converts a Celsius temperature to Fahrenheit", """
             if abs(celsiusToFahrenheit(0) - 32) > 0.001 { print("F1"); exit(1) }
             if abs(celsiusToFahrenheit(100) - 212) > 0.001 { print("F2"); exit(1) }
             if abs(celsiusToFahrenheit(37) - 98.6) > 0.001 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("binarySearch", "func binarySearch(_ a: [Int], _ target: Int) -> Int",
             "returns the index of target in the sorted array a, or -1 if not present", """
             if binarySearch([1, 3, 5, 7, 9], 5) != 2 { print("F1"); exit(1) }
             if binarySearch([1, 3, 5, 7, 9], 4) != -1 { print("F2"); exit(1) }
             if binarySearch([], 1) != -1 { print("F3"); exit(1) }
             if binarySearch([1, 2, 3], 1) != 0 { print("F4"); exit(1) }
             print("ALL_PASS")
             """),
        task("titleCase", "func titleCase(_ s: String) -> String",
             "capitalizes the first letter of each space-separated word, leaving the rest unchanged", """
             if titleCase("hello world") != "Hello World" { print("F1"); exit(1) }
             if titleCase("a b c") != "A B C" { print("F2"); exit(1) }
             if titleCase("") != "" { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("twoSum", "func twoSum(_ nums: [Int], _ target: Int) -> [Int]",
             "returns the indices [i, j] (i < j) of the two numbers that add up to target (exactly one solution exists)", """
             if twoSum([2, 7, 11, 15], 9) != [0, 1] { print("F1"); exit(1) }
             if twoSum([3, 2, 4], 6) != [1, 2] { print("F2"); exit(1) }
             if twoSum([3, 3], 6) != [0, 1] { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("removeDuplicates", "func removeDuplicates(_ a: [Int]) -> [Int]",
             "returns the array with duplicates removed, preserving first-occurrence order", """
             if removeDuplicates([1, 2, 2, 3, 1]) != [1, 2, 3] { print("F1"); exit(1) }
             if removeDuplicates([]) != [] { print("F2"); exit(1) }
             if removeDuplicates([5, 5, 5]) != [5] { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("digitSum", "func digitSum(_ n: Int) -> Int",
             "returns the sum of the decimal digits of n (n >= 0)", """
             if digitSum(123) != 6 { print("F1"); exit(1) }
             if digitSum(0) != 0 { print("F2"); exit(1) }
             if digitSum(99) != 18 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
    ]

    // ── HARD tier (LeetCode-medium): multi-step problems that genuinely REQUIRE reasoning/
    // planning, not recall. A 7B coder stumbles here, so this is the surface where CoT can show
    // LIFT (think > no-think). Use via HARD=1. trainHard / evalHard are disjoint.
    public static let trainHardTasks: [CodeTask] = [
        task("isBalanced", "func isBalanced(_ s: String) -> Bool",
             "returns true iff the brackets (), [], {} in s are correctly matched and nested", """
             if isBalanced("()[]{}") != true { print("F1"); exit(1) }
             if isBalanced("([)]") != false { print("F2"); exit(1) }
             if isBalanced("(([]){})") != true { print("F3"); exit(1) }
             if isBalanced("(]") != false { print("F4"); exit(1) }
             if isBalanced("") != true { print("F5"); exit(1) }
             print("ALL_PASS")
             """),
        task("romanToInt", "func romanToInt(_ s: String) -> Int",
             "converts a Roman numeral string to its integer value (handles subtractive forms like IV, IX, XL, CM)", """
             if romanToInt("III") != 3 { print("F1"); exit(1) }
             if romanToInt("IV") != 4 { print("F2"); exit(1) }
             if romanToInt("IX") != 9 { print("F3"); exit(1) }
             if romanToInt("LVIII") != 58 { print("F4"); exit(1) }
             if romanToInt("MCMXCIV") != 1994 { print("F5"); exit(1) }
             print("ALL_PASS")
             """),
        task("evalRPN", "func evalRPN(_ tokens: [String]) -> Int",
             "evaluates an arithmetic expression in Reverse Polish Notation (operators +,-,*,/ with integer truncation toward zero)", """
             if evalRPN(["2", "1", "+", "3", "*"]) != 9 { print("F1"); exit(1) }
             if evalRPN(["4", "13", "5", "/", "+"]) != 6 { print("F2"); exit(1) }
             if evalRPN(["2", "3", "-"]) != -1 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("mergeIntervals", "func mergeIntervals(_ intervals: [[Int]]) -> [[Int]]",
             "merges all overlapping intervals and returns them sorted by start", """
             if mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]]) != [[1, 6], [8, 10], [15, 18]] { print("F1"); exit(1) }
             if mergeIntervals([[1, 4], [4, 5]]) != [[1, 5]] { print("F2"); exit(1) }
             if mergeIntervals([[1, 4]]) != [[1, 4]] { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("editDistance", "func editDistance(_ a: String, _ b: String) -> Int",
             "returns the Levenshtein edit distance (min insertions/deletions/substitutions) between a and b", """
             if editDistance("horse", "ros") != 3 { print("F1"); exit(1) }
             if editDistance("intention", "execution") != 5 { print("F2"); exit(1) }
             if editDistance("", "abc") != 3 { print("F3"); exit(1) }
             if editDistance("abc", "abc") != 0 { print("F4"); exit(1) }
             print("ALL_PASS")
             """),
        task("spiralOrder", "func spiralOrder(_ matrix: [[Int]]) -> [Int]",
             "returns all elements of the matrix in clockwise spiral order starting top-left", """
             if spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) != [1, 2, 3, 6, 9, 8, 7, 4, 5] { print("F1"); exit(1) }
             if spiralOrder([[1, 2], [3, 4]]) != [1, 2, 4, 3] { print("F2"); exit(1) }
             print("ALL_PASS")
             """),
    ]

    public static let evalHardTasks: [CodeTask] = [
        task("wordBreak", "func wordBreak(_ s: String, _ dict: [String]) -> Bool",
             "returns true iff s can be segmented into a space-separated sequence of one or more words from dict (words reusable)", """
             if wordBreak("leetcode", ["leet", "code"]) != true { print("F1"); exit(1) }
             if wordBreak("applepenapple", ["apple", "pen"]) != true { print("F2"); exit(1) }
             if wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]) != false { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("longestPalindrome", "func longestPalindrome(_ s: String) -> String",
             "returns the longest palindromic substring of s (inputs chosen to have a unique answer)", """
             if longestPalindrome("cbbd") != "bb" { print("F1"); exit(1) }
             if longestPalindrome("racecar") != "racecar" { print("F2"); exit(1) }
             if longestPalindrome("a") != "a" { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("maxSubArray", "func maxSubArray(_ nums: [Int]) -> Int",
             "returns the largest sum of any contiguous non-empty subarray (Kadane's algorithm)", """
             if maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) != 6 { print("F1"); exit(1) }
             if maxSubArray([1]) != 1 { print("F2"); exit(1) }
             if maxSubArray([-1, -2, -3]) != -1 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
        task("coinChange", "func coinChange(_ coins: [Int], _ amount: Int) -> Int",
             "returns the fewest number of coins summing to amount, or -1 if impossible (amount 0 needs 0 coins)", """
             if coinChange([1, 2, 5], 11) != 3 { print("F1"); exit(1) }
             if coinChange([2], 3) != -1 { print("F2"); exit(1) }
             if coinChange([1], 0) != 0 { print("F3"); exit(1) }
             print("ALL_PASS")
             """),
    ]
}
