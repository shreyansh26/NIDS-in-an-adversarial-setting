/* ****shreyansh26**** */
#include <bits/stdc++.h>
#define ll          long long
#define pb          push_back
#define	endl		'\n'
#define pii         pair<ll int,ll int>
#define vi          vector<ll int>
#define all(a)      (a).begin(),(a).end()
#define F           first
#define S           second
#define sz(x)       (ll int)x.size()
#define hell        1000000007
#define rep(i,a,b)	for(ll int i=a;i<b;i++)
#define lbnd        lower_bound
#define ubnd        upper_bound
#define bs          binary_search
#define mp          make_pair
#define ios         ios_base::sync_with_stdio(false);  cin.tie(0);	cout.tie(0);
#define N  100005

using namespace std;

int main() {
	ios
	int TESTS=1;
	cin>>TESTS;
	int CASE = 1;
	while(TESTS--)	{
		int n;
		cin>>n;
		unordered_map<string, vector<string>> m;
		vector<vector<string>> v;
		for(int i=0; i<n; i++) {
			string s;
			cin>>s;
			for(int i=0; i<s.size(); i++) {
				m[s.substr(i,s.size()-i)].pb(s);
			}
		}

		for(auto i: m) {
			cout<<i.F<<endl;
		}
	}
	return 0;
}