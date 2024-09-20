import 'package:flutter/material.dart';
import 'package:text_summary/Paragraph_to_summary.dart';
import 'package:text_summary/home.dart';
import 'package:text_summary/url_to_summary.dart';

void main(){
  runApp(routes()
  
  );
}

class routes extends StatelessWidget{
 const routes({super.key});

  @override
  Widget build(BuildContext context){
    return MaterialApp(

      debugShowCheckedModeBanner: false,
      
      routes: {
        '/' :(context)=>  home(),
        '/UrlToSummary':(context)=> const UrlToSummary(),
        '/ParagraphToSummary':(context)=>const ParagraphToSummary(),


      },
    );
  }
}