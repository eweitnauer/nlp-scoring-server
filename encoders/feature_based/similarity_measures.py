from difflib import SequenceMatcher

#longest common substring
def longestCommonsubstring(str1, str2):
    match = SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
    return len(str1[match.a: match.a + match.size])/ (len(str1) + len(str2))

#longest common subsequence => contiguity requirement is dropped.
def longestCommonSubseq(str1 , str2):
    # find the length of the strings
    m = len(str1)
    n = len(str2)

    # declaring the array for storing the dp values
    lcs = [[None]*(n+1) for i in range(m+1)]

    # store lcs[m+1][n+1] using bottom up DP approach

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                lcs[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                lcs[i][j] = lcs[i-1][j-1]+1
            else:
                lcs[i][j] = max(lcs[i-1][j] , lcs[i][j-1])

    return lcs[m][n]/ (len(str1) + len(str2))
