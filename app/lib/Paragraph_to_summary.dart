import 'package:flutter/material.dart';

class ParagraphToSummary extends StatelessWidget{
  const ParagraphToSummary({super.key});

  @override

  Widget build(BuildContext context){
    return Scaffold(
      appBar: AppBar(
        title: const Text("Paragraph to summary"),
      ),
      body: const Center(
        child: Text("Paragraph to summary"),
      )
    );
  }
}