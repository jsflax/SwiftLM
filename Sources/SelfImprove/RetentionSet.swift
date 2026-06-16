import Foundation

/// A fixed, NON-transcript retention set (generic CS knowledge) used to detect
/// catastrophic forgetting: a promoted adapter must not raise loss here beyond budget.
/// Independent of the user's data on purpose.
public enum RetentionSet {
    private static func tmpl(_ user: String, _ asst: String) -> String {
        "<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n\(asst)<|im_end|>"
    }

    public static let examples: [String] = [
        tmpl("What is the time complexity of binary search, and why?",
             "Binary search runs in O(log n) time. Each comparison halves the remaining search interval, so after k steps the candidate range is n/2^k; it reaches size 1 when k = log2(n). It requires a sorted array."),
        tmpl("Explain the difference between a SQL INNER JOIN and a LEFT JOIN.",
             "An INNER JOIN returns only rows with a match in both tables. A LEFT JOIN returns every row from the left table, plus matching rows from the right; where there is no match, the right-side columns are NULL."),
        tmpl("Write a Python function that returns the factorial of n using recursion.",
             "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nThe base case n <= 1 returns 1; otherwise it multiplies n by the factorial of n-1."),
        tmpl("What is a hash map and what is its average lookup complexity?",
             "A hash map stores key/value pairs, using a hash function to map keys to buckets in an array. Average insert, lookup, and delete are O(1); worst case is O(n) when many keys collide into the same bucket."),
        tmpl("Describe the difference between git merge and git rebase.",
             "git merge combines two branches by creating a new merge commit, preserving the original history. git rebase replays your commits on top of another branch, producing a linear history but rewriting commit hashes."),
        tmpl("What does the HTTP status code 404 mean?",
             "404 Not Found means the server reached the request but could not find a resource matching the requested URL. It indicates the resource does not exist (or is hidden), not a server failure."),
        tmpl("Explain what a closure is in programming.",
             "A closure is a function together with a captured reference to the variables from the scope in which it was defined. It can read and mutate those variables even after that scope has returned, letting state persist across calls."),
        tmpl("How does quicksort work at a high level?",
             "Quicksort picks a pivot, partitions the array so elements less than the pivot come before it and greater elements after, then recursively sorts the two partitions. Average time is O(n log n); worst case O(n^2)."),
    ]
}
