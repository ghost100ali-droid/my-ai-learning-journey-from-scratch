#include<iostream>
#include<vector>
#include <cstdlib>
#include <ctime>
using namespace std;
// This algorithm implements Q-learning, a reinforcement learning technique used to find the optimal path in a 2D grid.
//  The agent begins at a starting position and aims to reach the end while avoiding obstacles. 
//  It can move in four directions: up, down, left, or right. A Q-table is maintained to store expected rewards (Q-values) 
//  for each state-action pair. During training, the agent selects actions using an epsilon-greedy policy, 
//  sometimes exploring randomly and sometimes exploiting the best-known action. After each move, 
//  the Q-value is updated based on the received reward and the maximum predicted future reward. 
//  Normal cells provide a small negative reward, while reaching the goal gives a high positive reward. 
//  After training, the agent can follow the highest Q-values to determine the optimal path to the target.
struct State {
    int row , col ;
    State(int row , int col ) : row(row), col(col){}
};
class QLearning{
    private:
        vector<vector<char>>grid;
        int GRIDROWS;      
        int GRIDCOLS; 
        State start;
        State end;  
        const int ACTIONS=4;
            double exploreRate = 1.0; 
            double minExploreRate = 0.03;
            double decay = 0.995;
        double alpha;
        double gamma;
        vector<vector<vector<double>>> Q;
        vector<vector<int>> rewards ;
        State move(State s , int action){
            State next=s;
            switch(action){
                case 0: if(s.row>0 && grid[s.row-1][s.col]!='#' )next.row--; break;
                case 1: if(s.row<GRIDROWS-1 && grid[s.row+1][s.col]!='#' )next.row++; break;
                case 2: if(s.col<GRIDCOLS-1 && grid[s.row][s.col+1]!='#' )next.col++; break;
                case 3: if(s.col>0 && grid[s.row][s.col-1]!='#' )next.col--; break;
            }
            return next;
        }
        int getRandomAction(){
            return rand()%ACTIONS;
        }
        void train(int episodes = 1){
            for ( int i=0 ;i < episodes; i++){
                State s = start;
                while ( true ){
                    int action;
                    if((double)rand() / RAND_MAX < exploreRate){
                        action = getRandomAction();
                    }else{
                        action =0;
                        for( int a =1 ; a<ACTIONS ; a++){
                            if(Q[s.row][s.col][action]<Q[s.row][s.col][a]){
                                action =a ;
                            }
                        }

                    }
                    State next = move(s,action);
                    int reward = rewards[next.row][next.col];

                    double maxQ = Q[next.row][next.col][0];
                    for( int a = 1 ; a < ACTIONS ; ++a ){
                        if (Q[next.row][next.col][a] > maxQ)
                            maxQ = Q[next.row][next.col][a];
                    }
                    Q[s.row][s.col][action] += alpha * (reward + gamma * maxQ - Q[s.row][s.col][action]);

                    s = next;
                    exploreRate = max(minExploreRate, exploreRate * decay);
                    if (reward == 100) break; 
                }
            }
        }
    
    public:
        QLearning(vector<vector<char>> grid , State start , State end ,  double alpha ,double gamma):grid(grid) , start(start) , end(end)  , alpha( alpha) , gamma(gamma){
            GRIDROWS = grid.size();
            GRIDCOLS = grid[0].size();
            Q = vector<vector<vector<double>>>(GRIDROWS, vector<vector<double>>(GRIDCOLS, vector<double>(ACTIONS, 0)));
            rewards = vector<vector<int>>(GRIDROWS , vector<int> (GRIDCOLS ,-1));
            rewards[end.row][end.col] = 100;
        }
        void findePath(){
            State s=start;
            vector<vector<char>> pathgrid=grid;
            pathgrid[s.row][s.col] = 'S';
            while( s.row!=end.row || s.col!=end.col ){
                int bestaction=0;
                for( int i =1; i< ACTIONS ; i++){
                    if(Q[ s.row][s.col][bestaction] < Q[s.row][s.col][i]){
                        bestaction = i;
                    }
                }
                s=move(s,bestaction);
                pathgrid[s.row][s.col] = 'x';
            }
            pathgrid[s.row][s.col] = 'E';
            printGrid(pathgrid);
        }
        void printGrid(vector<vector<char>> grid){
            cout<<"\n";
            for (auto& row : grid){
                for(auto& col : row){
                    cout<< col<<"\t";
                }
                cout<<"\n";
            }
        }
        void run(){
            train(1000);
            findePath();
        }
};
int main (){
srand(time(nullptr));

vector<vector<char>> grid ={
    {'.','.','.','#','.','.'},
    {'.','.','#','#','.','.'},
    {'.','.','#','.','.','.'},
    {'.','.','#','.','#','.'},
    {'.','.','#','.','#','.'},
    {'.','.','.','.','#','.'},
    {'.','.','.','.','#','.'},
    {'.','.','.','.','#','.'},
};
State start(0,0);
State end(7,5);
double alpha=0.1;
double gamma=0.8;
QLearning qlearing(grid, start , end , alpha , gamma );

qlearing.run();

}