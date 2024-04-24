package com.bomdemo.kimiapi.service;

import com.bomdemo.kimiapi.https.HttpsWithoutSSL;
import com.bomdemo.kimiapi.model.ChatContent;
import com.bomdemo.kimiapi.model.ResponseChoice;
import com.google.gson.Gson;
import org.apache.catalina.Host;
import org.apache.catalina.core.StandardHost;
import org.apache.http.HttpHost;
import org.apache.http.impl.client.CloseableHttpClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import static com.bomdemo.kimiapi.https.HttpsWithoutSSL.createHttpClientWithSSLVerificationDisabled;

@Service
public class ChatServiceImpl implements ChatService {
    @Value("${connection-server.url}")
    private String apiUrl;

    @Value("${connection-server.key}")
    private String authToken;

    @Override
    public ResponseChoice sentMessage(ChatContent message) {
        StringBuffer response = null;
        try {
            URL url = new URL(apiUrl);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Authorization", authToken);
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String inputLine;
                response = new StringBuffer();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();
                // 打印结果
                System.out.println(response.toString());
            } else {
                System.out.println("GET request failed");
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Gson gson = new Gson();
        ResponseChoice person = gson.fromJson(response.toString(), ResponseChoice.class);
        return person;
    }

}
