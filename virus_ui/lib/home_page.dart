import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  File? _image;
  String _prediction = "";
  double _confidence = 0;
  bool _isUploading = false;

  final ImagePicker _picker = ImagePicker();
  late AnimationController _confController;
  late AnimationController _glowController;

  @override
  void initState() {
    super.initState();

    _confController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    );

    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
      lowerBound: 0.6,
      upperBound: 1,
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _confController.dispose();
    _glowController.dispose();
    super.dispose();
  }

  Future<void> pickImage() async {
    try {
      final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _prediction = "";
          _confidence = 0;
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Image selection failed: $e')));
    }
  }

  Uri get _predictUri {
    try {
      if (Platform.isAndroid) {
        return Uri.parse('http://10.0.2.2:5000/predict');
      }
    } catch (_) {}
    return Uri.parse('http://localhost:5000/predict');
  }

  Future<void> uploadImage() async {
    if (_image == null) return;

    setState(() {
      _isUploading = true;
      _prediction = "";
      _confidence = 0;
    });

    try {
      final request = http.MultipartRequest('POST', _predictUri);
      request.files.add(
        await http.MultipartFile.fromPath('file', _image!.path),
      );

      final streamedResponse = await request.send();
      final responseBody = await streamedResponse.stream.bytesToString();

      if (streamedResponse.statusCode != 200) {
        throw Exception(
          'Server error ${streamedResponse.statusCode}: $responseBody',
        );
      }

      final data = jsonDecode(responseBody);

      final className = data['class']?.toString() ?? 'Unknown';
      double confidenceNum = 0.0;
      try {
        if (data['confidence'] is num) {
          confidenceNum = (data['confidence'] as num).toDouble();
        } else {
          confidenceNum =
              double.tryParse(data['confidence']?.toString() ?? '') ?? 0.0;
        }
      } catch (_) {
        confidenceNum = 0.0;
      }
      confidenceNum = confidenceNum.clamp(0.0, 1.0);

      if (!mounted) return;
      setState(() {
        _prediction = className;
        _confidence = confidenceNum;
      });

      _confController.forward(from: 0);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Prediction failed: $e')));
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  Widget buildPredictionCard() {
    if (_prediction.isEmpty) return const SizedBox.shrink();

    return FadeTransition(
      opacity: _confController,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Colors.cyan.shade700.withOpacity(0.7),
              Colors.blue.shade900.withOpacity(0.9),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(25),
          boxShadow: [
            BoxShadow(
              color: Colors.cyanAccent.withOpacity(0.6),
              blurRadius: 15,
              spreadRadius: 1,
            ),
          ],
          border: Border.all(
            color: Colors.cyanAccent.withOpacity(0.8),
            width: 1.5,
          ),
        ),
        child: Column(
          children: [
            Text(
              _prediction,
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.white,
                shadows: [Shadow(blurRadius: 5, color: Colors.cyanAccent)],
              ),
            ),
            const SizedBox(height: 15),
            AnimatedBuilder(
              animation: _confController,
              builder: (context, child) {
                return LinearProgressIndicator(
                  value: _confidence * _confController.value,
                  minHeight: 16,
                  backgroundColor: Colors.white12,
                  color: Colors.cyanAccent,
                );
              },
            ),
            const SizedBox(height: 10),
            Text(
              '${(_confidence * 100).toStringAsFixed(2)}%',
              style: const TextStyle(
                fontSize: 18,
                color: Colors.cyanAccent,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget buildGlowingButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
    required Color glowColor,
  }) {
    return ScaleTransition(
      scale: _glowController,
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, color: glowColor),
        label: Text(label, style: TextStyle(color: glowColor)),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.black87,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
            side: BorderSide(color: glowColor, width: 2),
          ),
          shadowColor: glowColor.withOpacity(0.8),
          elevation: 15,
          textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF070A1F),
      appBar: AppBar(
        title: const Text(
          '🧬 Virus Classifier',
          style: TextStyle(
            color: Colors.cyanAccent,
            fontWeight: FontWeight.bold,
            fontSize: 22,
            shadows: [Shadow(color: Colors.cyanAccent, blurRadius: 10)],
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.black87,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color(0xFF001024),
              const Color(0xFF001F4D),
              const Color(0xFF003366),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 30),
        child: Center(
          child: SingleChildScrollView(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(30),
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 12, sigmaY: 12),
                child: Container(
                  padding: const EdgeInsets.all(25),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.07),
                    borderRadius: BorderRadius.circular(30),
                    border: Border.all(
                      color: Colors.cyanAccent.withOpacity(0.3),
                      width: 1.5,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.cyanAccent.withOpacity(0.25),
                        blurRadius: 25,
                        spreadRadius: 1,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      AnimatedSwitcher(
                        duration: const Duration(milliseconds: 400),
                        child: _image == null
                            ? Icon(
                                Icons.image_outlined,
                                size: 160,
                                color: Colors.cyanAccent.withOpacity(0.7),
                                key: const ValueKey('placeholder'),
                              )
                            : ClipRRect(
                                borderRadius: BorderRadius.circular(25),
                                child: Image.file(
                                  _image!,
                                  height: 230,
                                  fit: BoxFit.cover,
                                  key: const ValueKey('image'),
                                ),
                              ),
                      ),
                      const SizedBox(height: 30),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          buildGlowingButton(
                            icon: Icons.photo_library,
                            label: 'Pick Image',
                            onPressed: _isUploading ? null : pickImage,
                            glowColor: Colors.cyanAccent,
                          ),
                          const SizedBox(width: 25),
                          buildGlowingButton(
                            icon: Icons.analytics,
                            label: _isUploading ? 'Predicting...' : 'Predict',
                            onPressed: (_image == null || _isUploading)
                                ? null
                                : uploadImage,
                            glowColor: Colors.lightBlueAccent,
                          ),
                        ],
                      ),
                      buildPredictionCard(),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
