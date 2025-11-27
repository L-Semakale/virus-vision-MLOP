import 'package:flutter/material.dart';
import 'home_page.dart';

void main() {
  runApp(const VirusApp());
}

class VirusApp extends StatelessWidget {
  const VirusApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Virus Image Classifier',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomePage(),
    );
  }
}
