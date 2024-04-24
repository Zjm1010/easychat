package com.bomdemo.kimiapi.model;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class MessageInfo {
    //只支持 system,user,assistant 其一
    private String role;
    //content 不得为空
    private String content;
}
