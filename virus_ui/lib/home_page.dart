import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  XFile? _imageFile;
  Uint8List? _imageBytes;

  String _prediction = "";
  double _confidence = 0.0;
  bool _isUploading = false;

  final ImagePicker _picker = ImagePicker();

  AnimationController? _confController;
  AnimationController? _glowController;

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
      upperBound: 1.0,
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _confController?.dispose();
    _glowController?.dispose();
    super.dispose();
  }

  /// 🔥 Correct Render.com API URL
  Uri get _predictUri {
    return Uri.parse('https://virus-vision-mlop.onrender.com/predict');
  }

  Future<void> pickImage() async {
    try {
      final picked = await _picker.pickImage(source: ImageSource.gallery);
      if (picked != null) {
        final bytes = await picked.readAsBytes();
        setState(() {
          _imageFile = picked;
          _imageBytes = bytes;
          _prediction = "";
          _confidence = 0.0;
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Image selection failed: $e')));
    }
  }

  Future<void> uploadImage() async {
    if (_imageBytes == null) return;

    setState(() {
      _isUploading = true;
      _prediction = "";
      _confidence = 0.0;
    });

    try {
      final request = http.MultipartRequest("POST", _predictUri);

      request.files.add(
        http.MultipartFile.fromBytes(
          "file",
          _imageBytes!,
          filename: _imageFile?.name ?? "image.jpg",
        ),
      );

      final streamed = await request.send();
      final response = await streamed.stream.bytesToString();

      if (streamed.statusCode != 200) {
        throw Exception("Server error ${streamed.statusCode}: $response");
      }

      final data = jsonDecode(response);

      /// UPDATED FIELD NAMES
      setState(() {
        _prediction = data["class_name"]?.toString() ?? "Unknown";
        _confidence = (data["confidence"] is num ? data["confidence"] : 0.0)
            .toDouble();
      });

      _confController?.forward(from: 0);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Prediction failed: $e')));
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  Widget buildPredictionCard() {
    if (_prediction.isEmpty) return const SizedBox.shrink();

    return FadeTransition(
      opacity: _confController ?? AlwaysStoppedAnimation<double>(1.0),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.cyan.shade800, Colors.blue.shade900],
          ),
          borderRadius: BorderRadius.circular(25),
          border: Border.all(color: Colors.cyanAccent, width: 2),
        ),
        child: Column(
          children: [
            Text(
              _prediction,
              style: const TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 15),
            LinearProgressIndicator(
              value: _confidence,
              minHeight: 14,
              color: Colors.cyanAccent,
              backgroundColor: Colors.white24,
            ),
            const SizedBox(height: 10),
            Text(
              '${(_confidence * 100).toStringAsFixed(1)}%',
              style: const TextStyle(
                fontSize: 18,
                color: Colors.cyanAccent,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget glowingButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
    required Color color,
  }) {
    return ScaleTransition(
      scale: _glowController ?? AlwaysStoppedAnimation<double>(1.0),
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, color: color),
        label: Text(label),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.black,
          foregroundColor: color,
          padding: const EdgeInsets.symmetric(horizontal: 26, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
            side: BorderSide(color: color),
          ),
          elevation: 10,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xff061227),
      appBar: AppBar(
        title: const Text(
          "🧬 Virus Classifier",
          style: TextStyle(color: Colors.cyanAccent),
        ),
        backgroundColor: Colors.black,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Center(
          child: Column(
            children: [
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 400),
                child: _imageBytes == null
                    ? Icon(
                        Icons.image_outlined,
                        size: 180,
                        color: Colors.cyanAccent.withOpacity(0.6),
                        key: const ValueKey("placeholder"),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: Image.memory(
                          _imageBytes!,
                          height: 230,
                          fit: BoxFit.cover,
                          key: const ValueKey("image"),
                        ),
                      ),
              ),
              const SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  glowingButton(
                    icon: Icons.photo,
                    label: "Pick Image",
                    onPressed: _isUploading ? null : pickImage,
                    color: Colors.cyanAccent,
                  ),
                  const SizedBox(width: 25),
                  glowingButton(
                    icon: Icons.analytics,
                    label: _isUploading ? "Predicting..." : "Predict",
                    onPressed: (_imageBytes == null || _isUploading)
                        ? null
                        : uploadImage,
                    color: Colors.lightBlueAccent,
                  ),
                ],
              ),
              buildPredictionCard(),
            ],
          ),
        ),
      ),
    );
  }
}
