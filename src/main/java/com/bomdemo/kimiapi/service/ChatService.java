package com.bomdemo.kimiapi.service;

import com.bomdemo.kimiapi.model.ChatContent;
import com.bomdemo.kimiapi.model.ResponseChoice;

public interface ChatService {


    ResponseChoice sentMessage(ChatContent message) throws Exception;
}
