import 'package:flutter/material.dart';

class UrlToSummary extends StatelessWidget{
 const UrlToSummary({super.key});

 
 @override
 Widget build(BuildContext context){
    return Scaffold(
      appBar: AppBar(title: const Text("Url to Summarize"),
      ),
      body: const Center(
        child: Text("Url To Summary"),
      ),
  );
  }
}