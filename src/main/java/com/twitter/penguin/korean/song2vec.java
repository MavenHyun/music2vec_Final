package com.twitter.penguin.korean;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import Jama.Matrix;
import com.twitter.penguin.korean.tokenizer.KoreanTokenizer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import scala.collection.Seq;
import edu.stanford.nlp.math.ArrayMath;


/**
 * Created by Maven Hyun on 2017-06-02.
 */
/**
 * Created by Maven Hyun on 2017-05-25.
 */

public class song2vec
{
    public int N = 0; /*number of features*/
    public int V = 0; /*vocabulary size*/
    public int C = 0; /*window size*/
    public int G = 0; /*number of negative samples*/
    public double L = 0.0; /*learning rate*/
    public int F = 0;
    public int total_V = 0; /*vocabulary size including duplicates*/

    public Map<String, String> song_id2title = new HashMap<String, String>();
    public Map<String, String> song_id2lyrics = new HashMap<String, String>();

    public Map<String, Integer> vocab_table = new HashMap<String, Integer>();
    public Map<String, Integer> vocab_num = new HashMap<String, Integer>();
    public Map<String, Double> vocab_neg = new HashMap<String, Double>();

    public ArrayList<String> context_words = new ArrayList<String>();
    public ArrayList<String> negative_words = new ArrayList<String>();
    public ArrayList<Integer> negative_index = new ArrayList<Integer>();

    public Matrix input2hidden_weight = Matrix.random(V, N);
    public Matrix hidden2output_weight = Matrix.random(N, V);

    public Matrix input = new Matrix(V,1);
    public Matrix hidden = new Matrix(N,1);
    public Matrix output = new Matrix(V,1);
    public Matrix error = new Matrix(V,1);

    song2vec(int context, int feature, double learn, int negative)
    {
        N = feature;
        C = context;
        L = learn;
        G = negative;
    }

    public void song_collect()
    {
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader("melon_final_lyrics.txt"));
            JSONObject jsonObject = (JSONObject) obj;
            song_id2lyrics.putAll( (Map<String, String>) obj);
        } catch (Exception e) {}

        try {
            Object obj = parser.parse(new FileReader("melon_final_songinfo.txt"));
            JSONObject jsonObject = (JSONObject) obj;
            song_id2title.putAll( (Map<String, String>) obj);
        } catch (Exception e) {}

        for (String id : song_id2title.keySet()) vocab_table.put(id.toString(), vocab_table.size());
    }

    public void song_tokenize()
    {
        for (String id : song_id2lyrics.keySet())
        {
            CharSequence normalized = TwitterKoreanProcessorJava.normalize(song_id2lyrics.get(id));
            Seq<KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava.tokenize(normalized);
            Seq<KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava.stem(tokens);

            for (KoreanTokenJava token : TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed))
            {
                String stem = token.getText();
                String pos = token.getPos().toString();
                if (pos.contains("Noun")||pos.contains("Adjective"))
                {
                    total_V++;
                    if (!vocab_num.containsKey(stem)) vocab_num.put(stem, 1);
                    else vocab_num.replace(stem, vocab_num.get(stem), vocab_num.get(stem) + 1);
                }
            }
        }
        song_filter();
        V = vocab_table.size();
        input2hidden_weight = Matrix.random(V, N);
        hidden2output_weight = Matrix.random(N, V);
    }

    public void song_filter()
    {
        for (String word : vocab_num.keySet())
        {
            if (vocab_num.get(word) > F)
            {
                if (!vocab_table.containsKey(word)) vocab_table.put(word, vocab_table.size());
            }
        }
    }

    public void song_negative()
    {
        double sum = 0.0;
        for (String word : vocab_num.keySet())
        {
            if (vocab_table.containsKey(word)) sum += Math.pow((double) vocab_num.get(word), 0.75);
        }

        for (String word: vocab_num.keySet())
        {
            if (vocab_table.containsKey(word))
            {
                double prob = (double) vocab_num.get(word) / sum;
                vocab_neg.put(word, prob);
            }
        }
    }

    public void song_preprocess()
    {
        long startTime = System.currentTimeMillis();
        song_collect();
        song_tokenize();
        song_negative();
        long elapsedTime = System.currentTimeMillis() - startTime;
        System.out.println("Song preprocessing " + elapsedTime + " ms");
    }

    public Matrix vectorize(String word)
    {
        double[] array = new double[V];
        array[vocab_table.get(word)] = 1.0;
        Matrix vector = new Matrix(array, 1);
        return vector.transpose();
    }

    public Matrix input_vector() /*project all context words(part of lyrics) through i2h weight matrix*/
    {
        Matrix vector = new Matrix(V, 1);
        for (String word : context_words) vector.plusEquals(vectorize(word));
        input = vector;
        return vector;
    }

    public Matrix hidden_vector()
    {
        long startTime = System.currentTimeMillis();
        Matrix vector = new Matrix(N, 1);
        vector = (input2hidden_weight.transpose()).times(input_vector());
        vector = vector.times(1/(double)C);
        hidden = vector;
        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Hidden " + elapsedTime + " ms"); */
        return vector;
    }

    public Matrix output_vector(String word)
    {
        negative_sampling(word);
        long startTime = System.currentTimeMillis();
        Matrix vector = new Matrix(V, 1);
        vector = (hidden2output_weight.transpose()).times(hidden_vector());
        double sum = 0;
        for (String neg : negative_words)
        {
            vector.set(vocab_table.get(neg), 0, Math.exp(vector.get(vocab_table.get(neg), 0)));
            sum += vector.get(vocab_table.get(neg), 0);
        }
        output = vector.times(1/sum);
        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Output " + elapsedTime + " ms"); */
        return vector.times(1/sum);
    }

    public Matrix output_vector()
    {
        long startTime = System.currentTimeMillis();
        Matrix vector = new Matrix(V, 1);
        vector = (hidden2output_weight.transpose()).times(hidden_vector());
        double sum = 0;
        for (int i = 0; i < V; i++)
        {
            vector.set(i, 0, Math.exp(vector.get(i, 0)));
            sum += vector.get(i, 0);
        }
        output = vector.times(1/sum);
        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Output " + elapsedTime + " ms"); */
        return vector.times(1/sum);
    }


    public void negative_sampling(String word) /*target word is the title + artist format string*/
    {
        negative_index.clear();
        long startTime = System.currentTimeMillis();
        double rand = new Random().nextDouble();
        int loop = G;
        while (loop != 0)
        {
            double cumulative = 0.0;
            for (String neg : vocab_neg.keySet())
            {
                cumulative += vocab_neg.get(neg);
                if (!negative_words.contains(neg)&&(cumulative >= rand)&&(!neg.contains(word)))
                {
                    negative_index.add(vocab_table.get(neg));
                    negative_words.add(neg);
                    loop--;
                    break;
                }
            }
        }
        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Negative Sampling " + elapsedTime + " ms"); */
    }

    public void update_hidden2output(String word, int column2)
    {
        long startTime = System.currentTimeMillis();
        error = output_vector(word).minusEquals(vectorize(word));
        for (String neg : negative_words)
        {
            Matrix replace = new Matrix(N, 1);
            int column = vocab_table.get(neg);
            replace = hidden2output_weight.getMatrix(0, N-1, column, column);
            replace.minusEquals(hidden.times(error.get(column, 0)).times(L));
            hidden2output_weight.setMatrix(0, N-1, column, column, replace);
        }
        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Update W' " + elapsedTime + " ms"); */
    }

    public void update_input2hidden(String word)
    {
        int col = vocab_table.get(word);
        update_hidden2output(word, col);
        long startTime = System.currentTimeMillis();
        Matrix replace = new Matrix(N,1);
        replace = input2hidden_weight.getMatrix(col, col, 0, N-1).transpose();
        for (int i = 0; i < N; i++)
        {
            double sum = 0.0;
            for (int index : negative_index) sum += hidden2output_weight.get(i, index) * error.get(index, 0);
            sum = sum * L / (double)C;
            replace.set(i, 0, sum);
        }
        input2hidden_weight.setMatrix(col, col, 0, N-1, replace.transpose());

        long elapsedTime = System.currentTimeMillis() - startTime;
        /* System.out.println("Update W " + elapsedTime + " ms"); */
    }


    public void train_feed(int iter)
    {
        for (int count = 0; count < iter; count++)
        {
            int sub_count = 0;
            for (String id : song_id2lyrics.keySet())
            {
                long startTime = System.currentTimeMillis();
                CharSequence normalized = TwitterKoreanProcessorJava.normalize(song_id2lyrics.get(id));
                Seq<KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava.tokenize(normalized);
                Seq<KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava.stem(tokens);
                ArrayList<String> lyrics_words = new ArrayList<String>();

                for (KoreanTokenJava token : TwitterKoreanProcessorJava.tokensToJavaKoreanTokenList(stemmed))
                {
                    String stem = token.getText();
                    String pos = token.getPos().toString();
                    if (pos.contains("Noun")||pos.contains("Adjective"))
                    {
                        if (vocab_table.containsKey(stem)) lyrics_words.add(stem);
                    }
                }

                for (int i = 0; i < lyrics_words.size() - C + 1; i++)
                {
                    for (int j = i; j < C + i; j++)
                    {
                        if (!context_words.contains(lyrics_words.get(j)))
                        {
                            context_words.add(lyrics_words.get(j));
                        }
                    }
                    update_input2hidden(id);
                    context_words.clear();
                    negative_words.clear();
                }
                sub_count++;
                long elapsedTime = System.currentTimeMillis() - startTime;
                System.out.println("Song feeding " + elapsedTime + " ms");
                System.out.println("남은 노래 수 :" + (song_id2lyrics.size() - sub_count));
            }
            System.out.println("남은 횟수: " + (iter - count));
            try { printout();} catch (Exception e) {}
        }
        test();
    }

    public double cosine_sim(Matrix a, Matrix b)
    {
        double operand1 = 0;
        double operand2 = 0;
        double operand3 = 0;
        for (int i = 0; i < a.getRowDimension(); i++)
        {
            operand1 += a.get(i, 0) * a.get(i, 0);
            operand2 += b.get(i, 0) * b.get(i, 0);
            operand3 += a.get(i, 0) * b.get(i, 0);
        }
        return operand3 / (Math.sqrt(operand1) * Math.sqrt(operand2));
    }

    public void printout() throws Exception
    {
        PrintWriter pw = new PrintWriter(new FileWriter("input2hidden.txt"));
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < N; j++) {
                pw.print(Double.toString(input2hidden_weight.get(i, j)) + "\t");
            }
            pw.println();
        }
        pw.close();
        pw = new PrintWriter(new FileWriter("hidden2output.txt"));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < V; j++) {
                pw.print(Double.toString(hidden2output_weight.get(i, j)) + "\t");
            }
            pw.println();
        }
        pw.close();
    }

    public void test()
    {
        ArrayList<String> input = new ArrayList<String>();
        ArrayList<String> input_p = new ArrayList<String>();
        try {
            FileReader f = new FileReader("input.txt");
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null) input.add(line);
        } catch (Exception e) {}
        for (String text : input) retrieve(text, "results.txt");

        try {
            FileReader f = new FileReader("input_p.txt");
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null) input_p.add(line);
        } catch (Exception e) {}
        for (String text : input_p) retrieve(text, "results_p.txt");

    }

    public void retrieve(String text, String file)
    {
        System.out.println("Vocabulary Size: " + V + " Window Size: " + C + " # of Neurons: " + N + " Learning Rate: " + L + " # of Negative Samples " + G);

        CharSequence normalized = TwitterKoreanProcessorJava.normalize(text);
        Seq<KoreanTokenizer.KoreanToken> tokens = TwitterKoreanProcessorJava.tokenize(normalized);
        Seq<KoreanTokenizer.KoreanToken> stemmed = TwitterKoreanProcessorJava.stem(tokens);

        context_words.clear();
        for (String token : TwitterKoreanProcessorJava.tokensToJavaStringList(stemmed))
        {
            if ((vocab_table.containsKey(token))&&(!context_words.contains(token))) context_words.add(token);
        }
        C = context_words.size();
        Map<String, Double> temp = new HashMap<String, Double>();
        Matrix result = new Matrix(V, 1);
        result = output_vector();
        for (String word : vocab_table.keySet())
        {
            temp.put(word, cosine_sim(result, vectorize(word)));
        }
        for (String word : temp.keySet())
        {
            System.out.print(String.format("%20f\t", temp.get(word)));
            System.out.println(text + "\t" + song_id2title.get(word));
        }
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(file));
            for (String word : temp.keySet())
            {
                pw.print(String.format("%20f\t", temp.get(word)));
                pw.println(text + "\t" + song_id2title.get(word));
            }
            pw.close();
        } catch (Exception e) {}
    }



}


