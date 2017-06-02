#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>

using namespace std;

int main(int argc, char* argv[]){
    //announcement issues
    vector<string> vec[13100];
    map<string, int> ZImap;
    ifstream ifs(argv[1]);  //import from Big5-ZhuYin.map
    ofstream ofs(argv[2]);  //output to ZhuYin-Big5.map
    string str;
    int count = 0;
    //read Big5-ZhuYin.map line by line
    while(getline(ifs, str)){
        //dealt with the same word issue
        int polycount = 0;
        string poly[5];
        //get the first 2 charactr as the Chinese word 
		string Big5 = str.substr(0,2);
        //check over every piece of ZhuYin and Chinese word
		for(int i = 0; i < str.size(); i++){
			bool flag = 0;
            //place the ZhuYin character into string and find any same 
			if(str[i] == ' '|| str[i] == '/'){	
				poly[polycount] = str.substr(i+1, 2);
                if(ZImap.count(poly[polycount]) == 0){
					ZImap[poly[polycount]] = count;
                    count++;
                }
                //if word is same as prev
				for(int j=0; j<polycount; j++)
					if(poly[polycount] == poly[j])
					    flag=1;
                //if it is a brand new word
				if(!flag)
					polycount ++;
			}
		}
        /*checkpoint
        cout << polycount << endl;
        */
        //push in all the ZhuYin word into map a.k.a linking new map
		for(int i=0; i < polycount; i++)	
			vec[ZImap[poly[i]]].push_back(Big5);
		if(ZImap.count(Big5) == 0){
			ZImap[Big5] = count;
			count ++;
		}
        //push in the word to the ZImap
		vec[ZImap[Big5]].push_back(Big5);
    }
    //outputing to new map file
	for(map<string ,int>::iterator iter = ZImap.begin(); iter != ZImap.end(); iter++){
		string character = iter -> first;
		int number = iter -> second;
        //print the header word first
        ofs << character << " ";
        //print the whole word vector one by one
		for(int i = 0; i< vec[number].size(); i++)
		{
			ofs << vec[number][i];
			if(i < vec[number].size() - 1)
				ofs << " ";
			else
				ofs << endl;
		}
	}
    //EOF
	ifs.close();
	ofs.close();
	return 0;
}