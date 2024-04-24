package com.bomdemo.kimiapi.model;

import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Service;

import java.util.List;

@Getter
@Setter
public class ChatContent {
    private String model = "moonshot-v1-8k";
    private List<MessageInfo> message;
    private double temperature = 0.3;
    private int max_tokens = 2048;
}