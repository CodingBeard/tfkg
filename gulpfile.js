const gulp = require('gulp');
const {args, img, js, font, scss, copyJs, watcher, browserSyncProxy} = require('./node_modules/cbfrontend/gulp-tasks');

gulp.task('default', gulp.series(args, gulp.parallel(js, font, img, scss), copyJs));
gulp.task('watch', gulp.parallel('default', watcher));
gulp.task('reload', gulp.series(browserSyncProxy, 'watch'));