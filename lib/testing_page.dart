import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

class TestingPage extends StatefulWidget {
  final bool isDarkMode;
  final Function(bool) onThemeChanged;

  const TestingPage({
    Key? key,
    required this.isDarkMode,
    required this.onThemeChanged,
  }) : super(key: key);

  @override
  State<TestingPage> createState() => _TestingPageState();
}

class _TestingPageState extends State<TestingPage> {
  List<Map<String, dynamic>> _randomExamples = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadRandomExamples();
  }

  /// Lee el CSV y carga 10 descripciones aleatorias
  Future<void> _loadRandomExamples() async {
    try {
      // Leer CSV desde assets
      final csvData =
          await rootBundle.loadString('assets/gym_exercise_dataset.csv');

      List<String> lines = csvData.split('\n');
      List<Map<String, dynamic>> examples = [];

      // Skip header (línea 0)
      for (int i = 1; i < lines.length; i++) {
        if (lines[i].isEmpty) continue;

        try {
          // Parsear línea del CSV
          List<String> parts = lines[i].split(',');
          if (parts.length >= 7) {
            String exercise = parts[0].replaceAll('"', '').trim();
            String description = parts[6].replaceAll('"', '').trim();

            if (description.isNotEmpty && exercise.isNotEmpty) {
              examples.add({
                'exercise': exercise,
                'description': description,
              });
            }
          }
        } catch (e) {
          continue;
        }
      }

      // Barajar y tomar 10 aleatorios
      if (examples.isNotEmpty) {
        examples.shuffle();
        List<Map<String, dynamic>> randomExamples = examples.take(10).toList();

        setState(() {
          _randomExamples = randomExamples;
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error loading random examples: $e');
      setState(() {
        _randomExamples = [];
        _isLoading = false;
      });
    }
  }

  /// Al clickear un ejemplo, navega a la main page y ejecuta la búsqueda
  void _onExampleTapped(String description) {
    // Pasar descripción a la main page
    Navigator.pop(context, description);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = widget.isDarkMode;

    return Scaffold(
      appBar: AppBar(
        title: const Text('RANDOM EXAMPLES'),
        backgroundColor: const Color(0xFF8B0000),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: IconButton(
              icon: Icon(
                widget.isDarkMode ? Icons.light_mode : Icons.dark_mode,
                color: Colors.white,
              ),
              onPressed: () => widget.onThemeChanged(!widget.isDarkMode),
            ),
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: isDark
                ? [Colors.grey[900]!, Colors.grey[800]!]
                : [Colors.grey[100]!, Colors.grey[50]!],
          ),
        ),
        child: _isLoading
            ? const Center(
                child: CircularProgressIndicator(
                  color: Color(0xFFFF0000),
                ),
              )
            : _randomExamples.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.error_outline,
                          size: 80,
                          color: Color(0xFFB22222),
                        ),
                        const SizedBox(height: 20),
                        const Text(
                          'No Examples Available',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 30),
                        ElevatedButton(
                          onPressed: () => Navigator.pop(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFFF0000),
                          ),
                          child: const Text(
                            'Back',
                            style: TextStyle(color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                  )
                : Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Refresh button
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            onPressed: _loadRandomExamples,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFFFF0000),
                              padding: const EdgeInsets.symmetric(vertical: 14),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            icon: const Icon(Icons.refresh, size: 24),
                            label: const Text(
                              'LOAD NEW EXAMPLES',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),
                        Text(
                          'Click on any example to search:',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: isDark ? Colors.grey[300] : Colors.grey[700],
                          ),
                        ),
                        const SizedBox(height: 15),
                        // Lista de ejemplos
                        Expanded(
                          child: ListView.builder(
                            itemCount: _randomExamples.length,
                            itemBuilder: (context, index) {
                              final example = _randomExamples[index];

                              return Padding(
                                padding: const EdgeInsets.only(bottom: 12.0),
                                child: Card(
                                  elevation: 2,
                                  color:
                                      isDark ? Colors.grey[850] : Colors.white,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: ListTile(
                                    contentPadding: const EdgeInsets.all(16),
                                    leading: CircleAvatar(
                                      backgroundColor: const Color(0xFFB22222),
                                      child: Text(
                                        '${index + 1}',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                    title: Text(
                                      example['description'],
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                        fontSize: 16,
                                      ),
                                    ),
                                    subtitle: Padding(
                                      padding: const EdgeInsets.only(top: 8.0),
                                      child: Text(
                                        example['exercise'],
                                        style: TextStyle(
                                          color: isDark
                                              ? Colors.grey[400]
                                              : Colors.grey[600],
                                        ),
                                      ),
                                    ),
                                    trailing: const Icon(
                                      Icons.arrow_forward_ios,
                                      color: Color(0xFFFF0000),
                                      size: 20,
                                    ),
                                    onTap: () => _onExampleTapped(
                                        example['description']),
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
      ),
    );
  }
}
