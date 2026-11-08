<?php
// JSON over stdin/stdout keeps corpus text and query strings out of shell commands.
error_reporting(E_ALL & ~E_DEPRECATED);
ini_set('display_errors', 'stderr');
require_once __DIR__ . '/vendor/xunsearch/lib/XS.class.php';

try {
    $request = json_decode(stream_get_contents(STDIN), true, 512, JSON_THROW_ON_ERROR);
    $action = $argv[1] ?? '';
    $xs = new XS($request['project_ini']);
    $database = $request['database'];
    if (!preg_match('/^env_[a-f0-9]{24}$/D', $database)) {
        throw new Exception('Invalid database identifier');
    }
    if ($action === 'index') {
        $index = $xs->index;
        $index->setDb($database);
        $index->beginRebuild();
        try {
            $handle = fopen($request['input'], 'r');
            if ($handle === false) throw new Exception('Cannot open index input');
            $count = 0;
            while (($line = fgets($handle)) !== false) {
                if (trim($line) === '') continue;
                $row = json_decode($line, true, 512, JSON_THROW_ON_ERROR);
                $doc = new XSDocument(array(
                    'doc_id' => $row['doc_id'], 'title' => $row['title'], 'content' => $row['content']
                ));
                $index->add($doc);
                $count++;
            }
            fclose($handle);
            if ($count !== $request['expected_n']) throw new Exception('Index input count mismatch');
            $index->endRebuild();
            for ($attempt = 0; $attempt < 60; $attempt++) {
                if ($index->flushIndex()) break;
                if ($attempt === 59) throw new Exception('Timed out flushing the index');
                sleep(1);
            }
        } catch (Throwable $error) {
            $index->stopRebuild();
            throw $error;
        }
        $result = array('success' => true, 'indexed_n' => $count);
    } elseif ($action === 'count') {
        $search = $xs->search;
        $search->setDb($database);
        $result = array('success' => true, 'count' => $search->getDbTotal());
    } elseif ($action === 'search') {
        $search = $xs->search;
        $search->setDb($database);
        $search->setWeightingScheme(0);
        $search->setFuzzy(true);
        $search->setAutoSynonyms(false);
        $limit = max(1, min(1000, intval($request['limit'])));
        $offset = max(0, intval($request['offset']));
        $documents = $search->setLimit($limit, $offset)->search($request['query'], false);
        $hits = array();
        foreach ($documents as $doc) {
            $hits[] = array('doc_id' => $doc->doc_id, 'score' => $doc->weight());
        }
        $result = array('success' => true, 'hits' => $hits);
    } else {
        throw new Exception('Unknown bridge action');
    }
    echo json_encode($result, JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR);
} catch (Throwable $error) {
    echo json_encode(array('success' => false, 'error' => $error->getMessage()));
    exit(1);
}
