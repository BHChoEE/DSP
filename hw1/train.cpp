#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include "hmm.h"

using namespace std;

int main(int argc, char**argv){
    //============ get input =============//
    //get model_init.txt
    HMM* hmm = new HMM;
    loadHMM(hmm, argv[2]);
    //get seq_model
    string line;
    ifstream seq_file;
    seq_file.open(argv[3]);
    vector<int*> data;
    while(seq_file >> line){
         int* tmp = new int[line.length()];
         for(int i = 0; i < line.length() ; ++i)
            tmp[i] = line.c_str()[i] - 'A';
         data.push_back(tmp);
    }
    seq_file.close();

//============ train ===============//
int iteration = atoi(argv[1]);
for(int iter = 0 ; iter < iteration ; iter++){
    double pi[6] = {0};
    double aij_epsilon[6][6] = {0};
    double aij_gamma[6] = {0};
    double bj_gamma[6][6] = {0};

    for(int line = 0 ; line < data.size(); line++){
    //every sample
    //============ get alpha ==========//
    double alpha[50][6] = {0};
    // alpha init
       for(int i = 0 ; i < 6 ; ++i)
            alpha[0][i] = hmm -> initial[i] * hmm -> observation[data[line][0]][i];
    // alpha induction
        for(int t = 0 ; t < 49 ; ++t){
            for(int j = 0 ; j < 6 ; ++j){
                double sum = 0;
                for(int i = 0 ; i < 6 ; ++i)
                    sum += alpha[t][i] * hmm -> transition[i][j];
                alpha[t+1][j] = sum * hmm -> observation[data[line][t+1]][j];
            }
        }
        /*
        for(int i = 0 ; i < 50 ; i++){
            for(int j = 0 ; j < 6 ;++j)
                cout << alpha[i][j] << " ";
            cout << endl;
        }*/
    //=========== get beta ===========//
    double beta[50][6] = {0};
    //beta init
        for(int i = 0 ; i < 6 ; ++i)
            beta[49][i] = 1;
    //beta induction
        for(int t = 48 ; t >= 0 ; --t){
            for(int i = 0 ; i < 6 ; ++i){
                double sum = 0;
                for(int j = 0 ; j < 6 ; ++j)
                    sum += hmm -> transition[i][j] * hmm -> observation[data[line][t+1]][j] * beta[t+1][j];
                beta[t][i] =sum;
            }
        }
        /*
        for(int i = 0 ; i < 50 ; ++i){
            for(int j = 0 ; j < 6 ; ++j)
                cout << beta[i][j] << " ";
            cout << endl;
        }*/
    //============= get gamma ===========//
    double gamma[50][6] = {0};
    double gamma_sum[50] = {0};
        for(int t = 0 ; t < 50 ; ++t)
            for(int i = 0 ; i < 6 ; ++i)
                gamma_sum[t] += alpha[t][i] * beta[t][i];
        for(int t = 0 ; t < 50 ; ++t)
            for(int i = 0 ; i < 6 ; ++i)
                gamma[t][i] = alpha[t][i]*beta[t][i] / gamma_sum[t];
        /*
        for(int i = 0 ; i < 50 ; ++i){
            for(int j = 0 ; j < 6 ; ++j)
                cout << gamma[i][j] << " ";
            cout << endl;
        }*/
    //============= get epsilon ==========//
    double epsilon[49][6][6] = {0};
    double epsilon_sum[49] = {0};
        for(int t = 0 ; t < 49 ; ++t){
            for(int i = 0 ; i < 6 ; ++i){
                for(int j = 0 ; j < 6 ; ++j){
                    epsilon_sum[t] += alpha[t][i] * hmm -> transition[i][j] * hmm -> observation[data[line][t+1]][j] * beta[t+1][j];
                }
            }
        }
        for(int t = 0 ; t < 49 ; ++t){
            for(int i = 0 ; i < 6 ; ++i){
                for(int j = 0 ; j < 6 ; ++j){
                    epsilon[t][i][j] = alpha[t][i] * hmm -> transition[i][j] * hmm -> observation[data[line][t+1]][j] * beta[t+1][j] / epsilon_sum[t];
                }
            }
        }/*
        for(int t = 0 ; t < 49 ; t++){
            for(int i = 0 ; i < 6 ; i++){
                for(int j = 0 ; j < 6 ; j++)
                    cout << epsilon[t][i][j];
                cout << endl;
            }
            cout << "\n\n";
        }*/
    //============= accumulate epsilon and gamma ============//
    double gamma_acc[6] = {0};
    double epsilon_acc[6][6] = {0};
    double gamma_ob_acc[6][6] = {0};
    for(int t = 0 ; t < 50 ; ++t)
        for(int i = 0 ; i < 6 ; ++i)
            gamma_acc[i] += gamma[t][i];
    
    /*
    for(int i = 0 ; i < 6 ; i++)
        cout << gamma_acc[i] << " ";
    cout << endl;
    */  
    for(int t = 0 ; t < 50 ; ++t)
        for(int i = 0 ; i < 6 ; ++i)
            for(int j = 0 ; j < 6 ; ++j)
                epsilon_acc[i][j] += epsilon[t][i][j];
    /*
    for(int i = 0 ; i < 6 ; ++i){
        for(int j = 0 ; j < 6 ; ++j)
            cout << epsilon_acc[i][j] << " ";
        cout << endl;
    }*/
    for(int t = 0 ; t < 50 ; ++t)
        for(int i = 0 ; i < 6 ; ++i){
            if(data[line][t] == 0)
                gamma_ob_acc[0][i] += gamma[t][i];
            else if(data[line][t] == 1)
                gamma_ob_acc[1][i] += gamma[t][i];
            else if(data[line][t] == 2)
                gamma_ob_acc[2][i] += gamma[t][i];
            else if(data[line][t] == 3)
                gamma_ob_acc[3][i] += gamma[t][i];
            else if(data[line][t] == 4)
                gamma_ob_acc[4][i] += gamma[t][i];
            else if(data[line][t] == 5)
                gamma_ob_acc[5][i] += gamma[t][i];
        }
    /*
    for(int i = 0 ; i < 6 ; ++i){
        for(int j = 0 ; j < 6 ; ++j)
            cout << gamma_ob_acc[i][j] << " ";
        cout << endl;
    }*/
    //=========== sum through all samples ===================//
    //pi
        for(int i = 0 ; i < 6 ; ++i)
            pi[i] += gamma[0][i];
    //aij
        for(int i = 0 ; i < 6 ; ++i)
            for(int j = 0 ; j < 6 ; ++j)
                aij_epsilon[i][j] += epsilon_acc[i][j];
        for(int i = 0 ; i < 6 ; ++i)
            aij_gamma[i] += gamma_acc[i];
    //bj
        for(int k = 0 ; k < 6 ; ++k)
            for(int i = 0 ; i < 6 ;++i)
                bj_gamma[k][i] += gamma_ob_acc[k][i];
    }
    //================ re-estimate model parameters ==================//
    //pi
    for(int i = 0 ; i < 6 ; ++i)
        hmm -> initial[i] = pi[i] / 10000;
    //aij
    for(int i = 0 ; i < 6 ; ++i)
        for(int j = 0 ; j < 6 ; ++j)
            hmm -> transition[i][j] = aij_epsilon[i][j] / aij_gamma[i];
    //bj
    for(int k = 0 ; k < 6 ; ++k)
        for(int i = 0 ; i < 6 ; ++i)
            hmm -> observation[k][i] = bj_gamma[k][i] / aij_gamma[i];
    cout << iter << endl;
}
    /*
    for(int i = 0 ; i < 6 ; ++i)
        cout << hmm -> initial[i] << " ";
    cout << endl;
    for(int i = 0 ; i < 6 ; ++i){
        for(int j = 0 ; j < 6 ; ++j)
            cout << hmm -> transition[i][j] << " ";
        cout << endl;
    }
    cout << endl;
    for(int i = 0 ; i < 6 ; ++i){
        for(int j = 0 ; j < 6 ; ++j)
            cout << hmm -> observation[i][j] << " ";
        cout << endl;
    }*/

    //============== dump HMM =====================//
    FILE* outfile;
    outfile = fopen(argv[4],"w");
    dumpHMM(outfile, hmm);
return 0;
}