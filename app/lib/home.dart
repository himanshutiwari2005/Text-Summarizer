import 'package:flutter/material.dart';
import 'package:text_summary/Paragraph_to_summary.dart';
import 'package:text_summary/url_to_summary.dart';



class home extends StatelessWidget{
  const home({super.key});

  @override  

  Widget build(BuildContext context){
    return MaterialApp(
    home: Scaffold(
      body: Container(
        alignment:  Alignment.center,

        decoration: const BoxDecoration(
          gradient:  LinearGradient(colors: [Color.fromARGB(255, 93, 192, 226),
          Color.fromRGBO(6, 233, 93, 0.933)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight )
        ),
        child:  Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children:  [
              const Text("Text Summarizer", style: TextStyle(fontSize: 30),),

              const SizedBox(height: 50,width: 20,),

              ElevatedButton(onPressed: (){
                Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) => const UrlToSummary(),
                    ),
                  );
                }, child:const Text(
                  "URL to Summarize ")
                  ),

              const SizedBox(height: 30,),

              ElevatedButton(onPressed: (){
                Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) => const ParagraphToSummary(),
                    ),
                  );
                }, 
                child: const Text(
                  "Paragraph to Summarize")
                  ),

            ],
            
          ),
          
        ),
      ) ,
      
    ),
    

  );
  }
}
