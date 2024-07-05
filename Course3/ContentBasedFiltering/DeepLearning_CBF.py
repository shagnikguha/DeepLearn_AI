import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

'''
top10_df

 	movie id 	num ratings 	ave rating 	title 	genres
0 	4993 	198 	4.1 	Lord of the Rings: The Fellowship of the Ring,... 	Adventure|Fantasy
1 	5952 	188 	4.0 	Lord of the Rings: The Two Towers, The 	Adventure|Fantasy
2 	7153 	185 	4.1 	Lord of the Rings: The Return of the King, The 	Action|Adventure|Drama|Fantasy
3 	4306 	170 	3.9 	Shrek 	Adventure|Animation|Children|Comedy|Fantasy|Ro...
4 	58559 	149 	4.2 	Dark Knight, The 	Action|Crime|Drama
5 	6539 	149 	3.8 	Pirates of the Caribbean: The Curse of the Bla... 	Action|Adventure|Comedy|Fantasy
6 	79132 	143 	4.1 	Inception 	Action|Crime|Drama|Mystery|Sci-Fi|Thriller
7 	6377 	141 	4.0 	Finding Nemo 	Adventure|Animation|Children|Comedy
8 	4886 	132 	3.9 	Monsters, Inc. 	Adventure|Animation|Children|Comedy|Fantasy
9 	7361 	131 	4.2 	Eternal Sunshine of the Spotless Mind 	Drama|Romance|Sci-Fi
'''

'''
bygenre_df

 	genre 	num movies 	ave rating/genre 	ratings per genre
0 	Action 	    321 	3.4 	            10377
1 	Adventure 	234 	3.4 	            8785
2 	Animation 	76 	    3.6 	            2588
3 	Children 	69 	    3.4 	            2472
4 	Comedy 	    326 	3.4 	            8911
5 	Crime 	    139 	3.5 	            4671
6 	Documentary 13 	    3.8 	            280
7 	Drama 	    342 	3.6 	            10201
8 	Fantasy 	124 	3.4 	            4468
9 	Horror 	    56 	    3.2 	            1345
10 	Mystery 	68 	    3.6 	            2497
11 	Romance 	151 	3.4 	            4468
12 	Sci-Fi 	    174 	3.4 	            5894
13 	Thriller 	245 	3.4 	            7659
'''