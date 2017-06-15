package com.twitter.penguin.korean;

import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Created by Maven Hyun on 2017-06-02.
 */
public class main
{ public static void main(String[] args) throws Exception
{
    song2vec maven = new song2vec(260, 50, 0.001, 20); /* C, N , learning_rate*/
    maven.song_preprocess();
    maven.train_feed(100);
    maven.printout();
    return;
}


}
