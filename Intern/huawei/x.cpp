// 给你一个长度为 m 的字符串 s，其中仅包含数字。另给你一个二维整数数组 queries，其中 queries[i] = [li, ri]。
// 对于每个 queries[i]，提取 子串 s[li..ri]，然后执行以下操作：
// 将子串中所有 非零数字 按照原始顺序连接起来，形成一个新的整数 x。如果没有非零数字，则 x = 0。
// 令 sum 为 x 中所有数字的 数字和 。答案为 x * sum。
// 返回一个整数数组 answer，其中 answer[i] 是第 i 个查询的答案。
// 由于答案可能非常大，请返回其对 109 + 7 取余数的结果。
// 子串 是字符串中的一个连续、非空 字符序列。

// 示例 1：
// 输入： s = "10203004", queries = [[0,7],[1,3],[4,6]]
// 输出： [12340, 4, 9]
// 解释：
// s[0..7] = "10203004"
// x = 1234
// sum = 1 + 2 + 3 + 4 = 10
// 因此，答案是 1234 * 10 = 12340。
// s[1..3] = "020"
// x = 2
// sum = 2
// 因此，答案是 2 * 2 = 4。
// s[4..6] = "300"
// x = 3
// sum = 3
// 因此，答案是 3 * 3 = 9。
// 示例 2：
// 输入： s = "1000", queries = [[0,3],[1,1]]
// 输出： [1, 0]
// 解释：
// s[0..3] = "1000"
// x = 1
// sum = 1
// 因此，答案是 1 * 1 = 1。
// s[1..1] = "0"
// x = 0
// sum = 0
// 因此，答案是 0 * 0 = 0。

// 示例 3：
// 输入： s = "9876543210", queries = [[0,9]]
// 输出： [444444137]
// 解释：
// s[0..9] = "9876543210"
// x = 987654321
// sum = 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 45
// 因此，答案是 987654321 * 45 = 44444444445。
// 返回结果为 44444444445 mod (10e9 + 7) = 444444137。

// 提示：
// 1 <= m == s.length <= 10e5
// s 仅由数字组成。
// 1 <= queries.length <= 10e	5
// queries[i] = [li, ri]
// 0 <= li <= ri < m


#include <iostream>

using namespace std;

vector<int> func(const string& sss, vector<pair<int, int>> qs) {
    vector<int> ans;
    for (auto q : qs) {
        string ss = sss.substr(q.first, q.second - q.first + 1);
        int sum = 0;
        string s = "";
        for (auto c : ss) {
            if (c != '0') {
                sum += (c - '0');
                s += c;
            }
        }
        int x = stoi(s);
        ans.push_back(x * sum);
    }

    return ans;
}

int main() {
    string sss = "10203004";
    vector<pair<int, int>> qs = {{0,7},{1,3},{4,6}};

    vector<int> ans = func(sss, qs);
    for (int i : ans) {
        cout << i << " ";
    }
}
